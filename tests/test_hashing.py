"""Tests for sha256 helpers."""

from __future__ import annotations

from scripps_workflow.hashing import sha256_file, sha256_text


class TestSha256:
    def test_text_known_vector(self):
        # SHA-256("") = e3b0c44...
        assert sha256_text("").startswith("e3b0c44")

    def test_text_round_trip(self):
        assert sha256_text("hello") == sha256_text("hello")
        assert sha256_text("hello") != sha256_text("world")

    def test_file_matches_text(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("hello")
        assert sha256_file(p) == sha256_text("hello")

    def test_file_streaming_handles_large_input(self, tmp_path):
        # Write something larger than the chunk size (1 MiB) to exercise
        # the streaming loop.
        p = tmp_path / "big.bin"
        p.write_bytes(b"a" * (3 * 1024 * 1024 + 17))
        digest = sha256_file(p)
        assert len(digest) == 64
        # And reproducible.
        assert digest == sha256_file(p)
