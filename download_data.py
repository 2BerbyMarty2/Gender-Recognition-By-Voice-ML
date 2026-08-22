"""
Re-download the datasets used by this project into ./data.

This project's ./data directory is intentionally excluded from git (see
.gitignore) because it holds ~1.6GB of raw audio. Run this script any time
you've deleted ./data locally to fetch everything again.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars) - see https://github.com/Kaggle/kagglehub#authenticate

Datasets:
- primaryobjects/voicegender              -> data/voice.csv (already tracked
                                              in git too, so this is just a
                                              convenience re-fetch)
- murtadhanajim/gender-recognition-by-voiceoriginal
                                           -> data/data/male, data/data/female
                                              (raw .wav recordings)
"""

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).parent / "data"


def _find(root: Path, name: str) -> Path | None:
    if (root / name).is_dir() or (root / name).is_file():
        return root / name
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def download_voice_csv() -> None:
    print("Downloading primaryobjects/voicegender ...")
    src_root = Path(kagglehub.dataset_download("primaryobjects/voicegender"))
    src_csv = _find(src_root, "voice.csv")
    if src_csv is None:
        raise FileNotFoundError(f"voice.csv not found under {src_root}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_csv, DATA_DIR / "voice.csv")
    print(f"  -> {DATA_DIR / 'voice.csv'}")


def download_raw_audio() -> None:
    print("Downloading murtadhanajim/gender-recognition-by-voiceoriginal ...")
    src_root = Path(
        kagglehub.dataset_download("murtadhanajim/gender-recognition-by-voiceoriginal")
    )
    dest = DATA_DIR / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for label in ("male", "female"):
        src_dir = _find(src_root, label)
        if src_dir is None:
            print(f"  WARNING: could not find a '{label}' folder under {src_root}")
            continue
        dest_dir = dest / label
        if dest_dir.exists():
            print(f"  {dest_dir} already exists, skipping")
            continue
        shutil.copytree(src_dir, dest_dir)
        print(f"  -> {dest_dir} ({len(list(dest_dir.iterdir()))} files)")


if __name__ == "__main__":
    download_voice_csv()
    download_raw_audio()
    print("Done.")
