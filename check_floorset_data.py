#!/usr/bin/env python3
"""Fast local integrity check for FloorSet Lite training files."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--files-per-worker", type=int, default=90)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--min-bytes", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root) / "floorset_lite"
    missing = []
    small = []
    present = 0
    total = args.workers * args.files_per_worker
    for worker in range(args.workers):
        worker_dir = root / f"worker_{worker}"
        if not worker_dir.is_dir():
            missing.append(str(worker_dir))
            continue
        for idx in range(args.files_per_worker):
            path = worker_dir / f"layouts_{idx * args.stride}.th"
            if not path.exists():
                missing.append(str(path))
                continue
            present += 1
            if path.stat().st_size < args.min_bytes:
                small.append(str(path))
    print(f"root={root}")
    print(f"expected_files={total}")
    print(f"present_files={present}")
    print(f"missing_count={len(missing)}")
    print(f"small_count={len(small)}")
    print(f"expected_samples={present * args.stride}")
    if missing:
        print("missing_examples:")
        for item in missing[:20]:
            print(f"  {item}")
    if small:
        print("small_examples:")
        for item in small[:20]:
            print(f"  {item}")
    return 1 if missing or small else 0


if __name__ == "__main__":
    raise SystemExit(main())
