# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cloud Storage helpers: credentials for GKE pods and parallel transfers.

Credentials
-----------
``resolve_credentials`` tries, in order:

1. ``ksa``: the pod's Kubernetes service account token, exchanged at the
   Security Token Service for a federated access token. This is the flow the
   Cloud Storage FUSE CSI driver uses, and it also works for pods that run
   with ``hostNetwork: true`` (there the GKE metadata server cannot identify
   the pod). The pod needs a projected service account token volume whose
   audience is the cluster's workload identity pool, e.g.::

       volumes:
       - name: gcs-ksa-token
         projected:
           sources:
           - serviceAccountToken:
               audience: <GCP_PROJECT>.svc.id.goog
               expirationSeconds: 3600
               path: token

   The token exchange audience is derived from the token's issuer
   (``identitynamespace:<pool>:<issuer>``) unless given explicitly.
2. ``default``: Application Default Credentials (metadata server, a key file
   named by ``GOOGLE_APPLICATION_CREDENTIALS``, or user credentials).

The returned object is a ``google.auth.credentials.Credentials`` usable by
``gcsfs`` (``token=creds``) and ``google.cloud.storage.Client``.

Transfers
---------
``upload_file`` / ``download_file`` split one object into ranged parts that
are transferred by a thread pool (XML multipart upload / ranged GETs). One
stream per file is usually far below the NIC rate of a TPU host; several
parts in flight per file are what make checkpoint-sized objects fast.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GcsUri",
    "parse_gs_uri",
    "resolve_credentials",
    "storage_client",
    "upload_file",
    "download_file",
    "delete_prefix",
]

DEFAULT_KSA_TOKEN_PATH = "/var/run/secrets/gcs-ksa/token"
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


@dataclass(frozen=True)
class GcsUri:
    bucket: str
    path: str  # object name or prefix, no leading slash

    def join(self, *parts: str) -> "GcsUri":
        path = "/".join(p.strip("/") for p in (self.path, *parts) if p.strip("/"))
        return GcsUri(self.bucket, path)

    def __str__(self) -> str:
        return f"gs://{self.bucket}/{self.path}"


def parse_gs_uri(uri: str) -> GcsUri:
    match = re.fullmatch(r"gs://([^/]+)/?(.*)", uri)
    if match is None:
        raise ValueError(f"not a gs:// URI: {uri!r}")
    return GcsUri(match.group(1), match.group(2).strip("/"))


def _jwt_claims(token: str) -> dict[str, Any]:
    """Unverified JWT payload (only used to read the issuer)."""
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def _ksa_credentials(token_path: str, audience: str | None, impersonate: str | None = None):
    from google.auth import identity_pool

    with open(token_path) as f:
        token = f.read().strip()
    if audience is None:
        claims = _jwt_claims(token)
        issuer = claims["iss"]
        match = re.search(r"/projects/([^/]+)/", issuer)
        if match is None:
            raise ValueError("cannot derive the workload identity pool from the token issuer")
        aud = claims.get("aud")
        pool = aud[0] if isinstance(aud, list) else aud
        if not pool or not str(pool).endswith(".svc.id.goog"):
            pool = f"{match.group(1)}.svc.id.goog"
        audience = f"identitynamespace:{pool}:{issuer}"
    info = {
        "type": "external_account",
        "audience": audience,
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {"file": token_path},
    }
    if impersonate:
        # The Kubernetes service account is bound to a Google service account
        # (``iam.gke.io/gcp-service-account``); act as that account.
        info["service_account_impersonation_url"] = (
            "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
            f"{impersonate}:generateAccessToken"
        )
    return identity_pool.Credentials.from_info(info, scopes=list(_SCOPES))


def resolve_credentials(
    mode: str = "auto",
    *,
    ksa_token_path: str = DEFAULT_KSA_TOKEN_PATH,
    audience: str | None = None,
    impersonate: str | None = None,
    probe_uri: str | None = None,
) -> tuple[Any, str, list[str]]:
    """Return ``(credentials, mode_used, notes)``.

    ``mode`` is ``auto``, ``ksa`` or ``default``. ``impersonate`` is the
    Google service account bound to the pod's Kubernetes service account, if
    any (``ksa`` then also tries the federated token through that account).
    With ``probe_uri`` (``gs://bucket/prefix``), each candidate must also
    write and delete a small object there before it is accepted.
    """
    import google.auth
    import google.auth.transport.requests

    notes: list[str] = []
    candidates: list[str] = []
    if mode in ("auto", "ksa") and os.path.exists(ksa_token_path):
        # With a bound Google service account, act as it first (what the GKE
        # metadata server does for pods on the pod network).
        candidates += ["ksa-impersonate", "ksa"] if impersonate else ["ksa"]
    elif mode == "ksa":
        raise FileNotFoundError(f"no service account token at {ksa_token_path}")
    if mode in ("auto", "default"):
        candidates.append("default")
    for name in candidates:
        try:
            if name == "ksa":
                creds = _ksa_credentials(ksa_token_path, audience)
            elif name == "ksa-impersonate":
                creds = _ksa_credentials(ksa_token_path, audience, impersonate)
            else:
                creds, _ = google.auth.default(scopes=list(_SCOPES))
            creds.refresh(google.auth.transport.requests.Request())
            if probe_uri:
                _probe(creds, parse_gs_uri(probe_uri), name)
            notes.append(f"{name}: ok")
            return creds, name, notes
        except Exception as exc:  # noqa: BLE001 - report and try the next one
            notes.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
    raise PermissionError("no usable Cloud Storage credentials: " + "; ".join(notes))


def storage_client(creds: Any, project: str | None = None):
    from google.cloud import storage

    return storage.Client(project=project or "_", credentials=creds)


def _probe(creds: Any, uri: GcsUri, tag: str) -> None:
    client = storage_client(creds)
    blob = client.bucket(uri.bucket).blob(uri.join(f".probe-{tag}-{os.getpid()}-{time.time_ns()}").path)
    blob.upload_from_string(b"probe")
    blob.delete()


def upload_file(
    client: Any,
    filename: str,
    uri: GcsUri,
    *,
    chunk_size: int = 64 << 20,
    max_workers: int = 8,
) -> int:
    """Upload one local file as one object with parallel parts; returns bytes."""
    from google.cloud.storage import transfer_manager

    size = os.path.getsize(filename)
    blob = client.bucket(uri.bucket).blob(uri.path)
    if size <= chunk_size or max_workers <= 1:
        blob.upload_from_filename(filename)
    else:
        transfer_manager.upload_chunks_concurrently(
            filename, blob, chunk_size=chunk_size, max_workers=max_workers,
            worker_type=transfer_manager.THREAD,
        )
    return size


def download_file(
    client: Any,
    uri: GcsUri,
    filename: str,
    *,
    chunk_size: int = 64 << 20,
    max_workers: int = 8,
) -> int:
    """Download one object with parallel ranged reads; returns bytes."""
    from google.cloud.storage import transfer_manager

    blob = client.bucket(uri.bucket).blob(uri.path)
    blob.reload()
    size = int(blob.size or 0)
    if size <= chunk_size or max_workers <= 1:
        blob.download_to_filename(filename)
    else:
        transfer_manager.download_chunks_concurrently(
            blob, filename, chunk_size=chunk_size, max_workers=max_workers,
            worker_type=transfer_manager.THREAD,
        )
    return size


def delete_prefix(client: Any, uri: GcsUri) -> int:
    """Delete every object under ``uri`` (a prefix); returns the count."""
    bucket = client.bucket(uri.bucket)
    prefix = uri.path.rstrip("/") + "/"
    blobs = list(client.list_blobs(uri.bucket, prefix=prefix))
    for i in range(0, len(blobs), 100):
        with client.batch():
            for blob in blobs[i : i + 100]:
                bucket.delete_blob(blob.name)
    return len(blobs)
