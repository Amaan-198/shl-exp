from pathlib import Path
import os

from huggingface_hub import snapshot_download
from src import config


def main() -> None:
    # Use the same HF env as the rest of your app, but force ONLINE for this script
    os.environ.update(config.HF_ENV_VARS)
    os.environ["HF_HUB_OFFLINE"] = "0"  # allow downloads just for this script

    cache_root = Path(os.environ.get("TRANSFORMERS_CACHE", str(config.MODELS_DIR))).resolve()
    print(f"Using TRANSFORMERS_CACHE: {cache_root}")

    def fetch(repo_id: str) -> str:
        print(f"\nDownloading repo: {repo_id}")
        local_path = snapshot_download(
            repo_id=repo_id,
            local_files_only=False,   # we WANT network here
            resume_download=True,
        )
        print(f"Cached at: {local_path}")

        # Quick sanity check for config.json
        cfg = Path(local_path) / "config.json"
        if cfg.exists():
            print(f"  ✅ Found config.json at: {cfg}")
        else:
            print(f"  ⚠️  WARNING: config.json NOT found in: {local_path}")
        return local_path

    # Primary reranker from config
    primary_repo = config.BGE_RERANKER_MODEL  # "BAAI/bge-reranker-base"
    primary_path = fetch(primary_repo)

    # Fallback reranker
    fallback_repo = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fallback_path = fetch(fallback_repo)

    print("\nSummary:")
    print(f"  Primary reranker cached at:  {primary_path}")
    print(f"  Fallback reranker cached at: {fallback_path}")
    print("\n✅ Finished downloading reranker models for offline use.")


if __name__ == "__main__":
    main()
