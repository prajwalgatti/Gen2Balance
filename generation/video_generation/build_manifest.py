"""Build a single generation manifest from the per-class prompt files.

Replaces the old manual step of concatenating every <label_id>.json into one
global file. The manifest is a flat, ordered list so SLURM array jobs can slice
it by global index, while each entry carries a `name` = "<label_id>_<sample_idx>"
so videos are saved with their final name (no post-hoc renaming):

    [ {"name": "0_0", "label_id": 0, "sample_idx": 0, "prompt": "..."}, ... ]

Optionally also writes the release-facing prompts.json ({name: prompt}).
"""
import os
import glob
import json
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Concatenate per-class prompt files into one generation manifest.")
    p.add_argument("--prompts_dir", required=True,
                   help="directory of <label_id>.json prompt lists (text_prompt_generation output)")
    p.add_argument("--output", default="video_prompts.json",
                   help="output manifest path")
    p.add_argument("--mapping_output", default=None,
                   help="optional release mapping {name: prompt} (e.g. prompts.json)")
    return p.parse_args()


def main():
    args = parse_args()

    files = glob.glob(os.path.join(args.prompts_dir, "*.json"))
    label_ids = sorted((os.path.splitext(os.path.basename(f))[0] for f in files), key=int)

    manifest = []
    for label_id in label_ids:
        prompts = json.load(open(os.path.join(args.prompts_dir, f"{label_id}.json")))
        for sample_idx, prompt in enumerate(prompts):
            manifest.append({
                "name": f"{label_id}_{sample_idx}",
                "label_id": int(label_id),
                "sample_idx": sample_idx,
                "prompt": prompt,
            })

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} prompts across {len(label_ids)} classes -> {args.output}")

    if args.mapping_output:
        mapping = {e["name"]: e["prompt"] for e in manifest}
        with open(args.mapping_output, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"Wrote release mapping ({len(mapping)} entries) -> {args.mapping_output}")


if __name__ == "__main__":
    main()
