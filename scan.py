"""Turn photos in ~/skyphot into bosses. Use --watch to keep picking up new photos."""
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from pipeline import backfill_faces, process_image

load_dotenv(Path(__file__).parent / ".env")
PHOTO_DIR = Path.home() / "skyphot"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}

seen = set()  # (path, mtime) already handled this session


def scan() -> int:
    made = 0
    for path in sorted(PHOTO_DIR.iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
            continue
        key = (path, path.stat().st_mtime)
        if key in seen:
            continue
        seen.add(key)
        try:
            boss = process_image(path.read_bytes(), source=path.name)
        except Exception as e:  # one bad file shouldn't stop the scan
            print(f"  ! {path.name}: {e}")
            continue
        if boss:
            made += 1
            print(f"  + {path.name} -> {boss['name']}, {boss['title']}")
        else:
            print(f"  - {path.name}: already a boss")
    return made


if __name__ == "__main__":
    PHOTO_DIR.mkdir(exist_ok=True)
    if n := backfill_faces():
        print(f"Added face data to {n} existing boss(es).")
    print(f"Scanning {PHOTO_DIR} ...")
    print(f"{scan()} new boss(es).")
    if "--watch" in sys.argv:
        print("Watching for new photos (Ctrl+C to stop)...")
        while True:
            time.sleep(3)
            scan()
