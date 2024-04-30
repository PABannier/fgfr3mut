import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "PABannier/fgfr3mut"

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(REPO_ID, local_dir=out_dir, repo_type="dataset")
    print("Done.")
