# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the Cloud Storage helpers that need no network."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from unittest import mock

from torchtitan.experiments.tpu.checkpointing import gcs

ISSUER = "https://container.googleapis.com/v1/projects/example-project/locations/region/clusters/example"


def _token(claims: dict) -> str:
    segments = [
        base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=") for obj in ({"alg": "RS256"}, claims)
    ]
    return ".".join([*segments, "signature"])


def _has_google_auth() -> bool:
    try:
        import google.auth.transport.requests  # noqa: F401
        from google.auth import identity_pool  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class UriTest(unittest.TestCase):
    def test_parse_and_join(self):
        uri = gcs.parse_gs_uri("gs://bucket/a/b/")
        self.assertEqual((uri.bucket, uri.path), ("bucket", "a/b"))
        self.assertEqual(str(uri.join("c", "/d/")), "gs://bucket/a/b/c/d")
        self.assertEqual(str(gcs.parse_gs_uri("gs://bucket").join("x")), "gs://bucket/x")
        with self.assertRaises(ValueError):
            gcs.parse_gs_uri("/local/path")

    def test_bench_transfers_use_this_module(self):
        """dcp_io_bench's Cloud Storage transfers come from gcs.py, imported directly."""
        import sys
        import types

        from torchtitan.experiments.tpu.checkpointing import dcp_io_bench

        upload, download, root = dcp_io_bench._transfer_fns(types.SimpleNamespace(gs_root="gs://bucket/bench"), "run")
        self.assertIs(upload, gcs.upload_file)
        self.assertIs(download, gcs.download_file)
        self.assertEqual(root, gcs.parse_gs_uri("gs://bucket/bench/run"))
        self.assertNotIn("torchtitan_ckpt_gcs", sys.modules)


class CredentialsTest(unittest.TestCase):
    def test_jwt_claims(self):
        claims = {"iss": ISSUER, "aud": ["example-project.svc.id.goog"], "sub": "system:serviceaccount:ns:sa"}
        self.assertEqual(gcs._jwt_claims(_token(claims)), claims)

    @unittest.skipUnless(_has_google_auth(), "needs google-auth")
    def test_workload_identity_audience(self):
        cases = {
            "pool from aud": (["example-project.svc.id.goog"], "example-project.svc.id.goog"),
            "pool from issuer": ("https://kubernetes.default.svc", "example-project.svc.id.goog"),
        }
        for name, (aud, pool) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "token")
                with open(path, "w") as f:
                    f.write(_token({"iss": ISSUER, "aud": aud}))
                with mock.patch("google.auth.identity_pool.Credentials.from_info") as from_info:
                    gcs._ksa_credentials(path, None, impersonate="trainer@example-project.iam.gserviceaccount.com")
                info = from_info.call_args.args[0]
                self.assertEqual(info["audience"], f"identitynamespace:{pool}:{ISSUER}")
                self.assertEqual(info["credential_source"], {"file": path})
                self.assertTrue(info["service_account_impersonation_url"].endswith(
                    "/trainer@example-project.iam.gserviceaccount.com:generateAccessToken"))

    @unittest.skipUnless(_has_google_auth(), "needs google-auth")
    def test_ksa_mode_needs_a_token(self):
        with self.assertRaises(FileNotFoundError):
            gcs.resolve_credentials("ksa", ksa_token_path="/nonexistent/token")


if __name__ == "__main__":
    unittest.main()
