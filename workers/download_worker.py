"""
workers/download_worker.py
--------------------------
Micro-Worker #10 — Download & Cache Manager

Responsibility: Check whether a model is already cached locally, report
its disk size, and trigger a pre-download (snapshot_download) if needed.
Uses huggingface-hub only — no torch, no transformers weight loading.

RAM budget: ~50 MB (hub metadata calls only)
"""

from __future__ import annotations

import os
import traceback
from typing import Optional
from workers.base_worker import BaseWorker, WorkerResult


class DownloadWorker(BaseWorker):
    """
    Manages model cache discovery and pre-download.

    run() kwargs
    ------------
    action        : str  — "check" | "download" | "size"
    hf_repo       : str  — HuggingFace repo id, e.g. "Qwen/Qwen2.5-3B-Instruct"
    cache_dir     : str  — optional cache path override
    token         : str  — optional HF auth token for gated models

    WorkerResult.data  (action="check")
    -----------------
    dict:
      is_cached     : bool
      cache_path    : str | None
      size_gb       : float

    WorkerResult.data  (action="download")
    -----------------
    dict:
      local_dir     : str
      status        : "downloaded" | "already_cached"

    WorkerResult.data  (action="size")
    -----------------
    dict:
      size_gb       : float   — estimated disk size
      files         : int     — number of files
    """

    worker_id = "download"
    ram_budget_mb = 50.0
    requires_model = False

    def run(
        self,
        action: str = "check",
        hf_repo: str = "",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs,
    ) -> WorkerResult:
        if not hf_repo:
            return self._err("DownloadWorker.run() requires 'hf_repo' kwarg.")

        try:
            if action == "check":
                return self._check(hf_repo, cache_dir, token)
            elif action == "download":
                return self._download(hf_repo, cache_dir, token)
            elif action == "size":
                return self._estimate_size(hf_repo, token)
            else:
                return self._err(f"Unknown DownloadWorker action: '{action}'")
        except Exception as exc:
            return self._err(f"DownloadWorker error: {exc}\n{traceback.format_exc()}")

    # ── Internal methods ──────────────────────────────────────────────────

    def _check(self, hf_repo: str, cache_dir: Optional[str], token: Optional[str]) -> WorkerResult:
        """Check if the model exists in the HF cache."""
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir(cache_dir=cache_dir)
            for repo in cache.repos:
                if repo.repo_id == hf_repo:
                    size_gb = repo.size_on_disk / (1024 ** 3)
                    return self._ok(
                        {"is_cached": True, "cache_path": str(repo.repo_path), "size_gb": round(size_gb, 2)},
                        ram_mb=20.0,
                    )
        except Exception:
            pass

        # Fallback: check default HF cache directory manually
        default_cache = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        # HF stores repos as "models--{owner}--{name}"
        repo_dir_name = "models--" + hf_repo.replace("/", "--")
        repo_path = os.path.join(default_cache, repo_dir_name)
        if os.path.isdir(repo_path):
            size_bytes = sum(
                f.stat().st_size for f in os.scandir(repo_path) if f.is_file()
            )
            return self._ok(
                {"is_cached": True, "cache_path": repo_path, "size_gb": round(size_bytes / (1024 ** 3), 2)},
                ram_mb=5.0,
            )

        return self._ok({"is_cached": False, "cache_path": None, "size_gb": 0.0}, ram_mb=5.0)

    def _download(self, hf_repo: str, cache_dir: Optional[str], token: Optional[str]) -> WorkerResult:
        """Pre-download the model snapshot to local cache."""
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=hf_repo,
                cache_dir=cache_dir,
                token=token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            return self._ok({"local_dir": local_dir, "status": "downloaded"}, ram_mb=30.0)
        except Exception as exc:
            return self._err(f"Snapshot download failed for '{hf_repo}': {exc}")

    def _estimate_size(self, hf_repo: str, token: Optional[str]) -> WorkerResult:
        """Estimate model size from HF Hub metadata (no download required)."""
        try:
            from huggingface_hub import list_repo_files, hf_hub_url
            import requests

            files = list(list_repo_files(hf_repo, token=token))
            total_bytes = 0
            count = 0
            for fname in files:
                if fname.endswith((".safetensors", ".bin", ".pt", ".gguf")):
                    try:
                        url = hf_hub_url(hf_repo, fname, token=token)
                        r = requests.head(url, timeout=5, allow_redirects=True)
                        cl = r.headers.get("Content-Length")
                        if cl:
                            total_bytes += int(cl)
                            count += 1
                    except Exception:
                        pass

            size_gb = total_bytes / (1024 ** 3)
            return self._ok({"size_gb": round(size_gb, 2), "files": count}, ram_mb=10.0)
        except Exception as exc:
            return self._err(f"Size estimation failed: {exc}")
