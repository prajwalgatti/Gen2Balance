"""Stage 1 of text-prompt generation: action-profile generation.

For each class we sample a few real training videos as in-context exemplars,
upload them to Gemini, and ask it to produce a structured "action profile"
(definition / key visual elements / common mistakes). We generate several
profiles per class, each seeded by a different random sample of exemplars.

Exemplar videos are read directly from the benchmark's real train.csv (paths are
relative to --data_root, matching the rest of this repo), so no hand-made
label->videos mapping is needed.

Set the API key via the GOOGLE_API_KEY environment variable.
"""
import os
import json
import time
import random
import logging
import argparse
import concurrent.futures
from collections import defaultdict

from tqdm import tqdm
from google import genai

from prompts import generate_action_definition_prompt, extract_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Generate per-class action profiles with Gemini.")
    p.add_argument("--label_map", required=True,
                   help="data/annotations/<benchmark>/misc/label_id_to_label.json")
    p.add_argument("--train_csv", required=True,
                   help="data/annotations/<benchmark>/real/train.csv (source of exemplar videos)")
    p.add_argument("--data_root", default="",
                   help="root prepended to the relative video paths in train_csv")
    p.add_argument("--output_dir", required=True,
                   help="where label_<id>_gen_<idx>.json profiles are written")
    p.add_argument("--num_profiles", type=int, default=5,
                   help="number of action profiles per class")
    p.add_argument("--num_exemplars", type=int, default=5,
                   help="number of in-context exemplar videos per profile")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--seed", type=int, default=128)
    p.add_argument("--max_workers", type=int, default=8,
                   help="concurrent upload->generate->delete cycles (lower if you hit rate limits)")
    return p.parse_args()


def build_label_to_videos(train_csv, data_root):
    """{label_id (str): [absolute video paths]} derived from the real train.csv."""
    mapping = defaultdict(list)
    with open(train_csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            mapping[label].append(os.path.join(data_root, path))
    return mapping


def upload_video_and_wait(client, path):
    """Upload a single video and poll until it is ACTIVE."""
    video_file = client.files.upload(file=path)
    try:
        while video_file.state == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
        if video_file.state != "ACTIVE":
            raise RuntimeError(f"File {video_file.name} failed state: {video_file.state}")
        return video_file
    except Exception:
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass
        raise


def process_task(client, model, task):
    """Upload exemplars -> generate one action profile -> save -> cleanup."""
    label_id, label, exemplar_paths, idx, save_dir = task
    save_path = os.path.join(save_dir, f"label_{label_id}_gen_{idx}.json")
    if os.path.exists(save_path):
        return

    uploaded = []
    try:
        for path in exemplar_paths:
            uploaded.append(upload_video_and_wait(client, path))

        contents = uploaded + [generate_action_definition_prompt(label)]

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=contents)
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        if response is None:
            raise RuntimeError("Failed to generate content after retries")

        try:
            profile = extract_json(response.text)
            profile["label_id"] = label_id
            profile["label"] = label
            profile["videos_selected"] = exemplar_paths
            with open(save_path, "w") as f:
                json.dump(profile, f, indent=4)
        except json.JSONDecodeError:
            with open(save_path.replace(".json", "_raw.txt"), "w") as f:
                f.write(response.text)
    except Exception as e:
        logging.error(f"Error on label {label_id} gen {idx}: {e}")
    finally:
        for v in uploaded:
            try:
                client.files.delete(name=v.name)
            except Exception:
                pass


def main():
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set the GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    label_id_2_label = json.load(open(args.label_map))
    label_id_2_videos = build_label_to_videos(args.train_csv, args.data_root)

    # Pre-compute all jobs sequentially so the random exemplar sampling is
    # deterministic w.r.t. --seed, independent of how many workers run.
    logging.info("Planning jobs (deterministic exemplar sampling)...")
    jobs = []
    for label_id, label in label_id_2_label.items():
        videos = label_id_2_videos.get(label_id, [])
        if len(videos) < args.num_exemplars:
            continue
        for idx in range(args.num_profiles):
            if os.path.exists(os.path.join(args.output_dir, f"label_{label_id}_gen_{idx}.json")):
                continue
            exemplars = random.sample(videos, args.num_exemplars)
            jobs.append((label_id, label, exemplars, idx, args.output_dir))

    logging.info(f"Running {len(jobs)} jobs with {args.max_workers} workers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(lambda t: process_task(client, args.model, t), jobs),
                  total=len(jobs)))


if __name__ == "__main__":
    main()
