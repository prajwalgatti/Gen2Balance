"""Check that the videos referenced by a benchmark's annotation CSVs exist.

Scans every CSV under data/annotations/<benchmark>/ (train/test of all splits),
collects the unique relative video paths, and reports which are missing under
--data_root.

    python data/verify_videos.py --data_root /path/to/data --benchmark K100-LT
"""
import os
import glob
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Verify benchmark videos exist under DATA_ROOT.")
    p.add_argument("--data_root", required=True, help="root the CSV paths are relative to")
    p.add_argument("--benchmark", required=True,
                   choices=["K100-LT", "UCF-LT", "RareAct-K100-LT"])
    p.add_argument("--annotations_dir", default="data/annotations",
                   help="path to the annotations/ directory")
    p.add_argument("--show", type=int, default=10, help="how many missing examples to print")
    return p.parse_args()


def main():
    args = parse_args()
    bench_dir = os.path.join(args.annotations_dir, args.benchmark)
    csvs = sorted(glob.glob(os.path.join(bench_dir, "**", "*.csv"), recursive=True))
    if not csvs:
        raise SystemExit(f"No CSVs found under {bench_dir}")

    rel_paths = set()
    for csv in csvs:
        with open(csv) as f:
            for line in f:
                line = line.strip()
                if line:
                    rel_paths.add(line.rsplit(" ", 1)[0])  # drop the trailing label

    missing = [p for p in sorted(rel_paths)
               if not os.path.exists(os.path.join(args.data_root, p))]

    print(f"{args.benchmark}: {len(csvs)} CSV files, {len(rel_paths)} unique clips referenced")
    print(f"present: {len(rel_paths) - len(missing)}   missing: {len(missing)}")
    for p in missing[:args.show]:
        print(f"  MISSING: {p}")
    if len(missing) > args.show:
        print(f"  ... and {len(missing) - args.show} more")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
