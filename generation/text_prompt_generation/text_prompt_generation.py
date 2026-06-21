"""Stage 2 of text-prompt generation: per-class text prompts.

For each class we load its action profiles (stage 1) and repeatedly ask Gemini
for batches of diverse text-to-video prompts until we have the requested number
of unique prompts. Output is one file per class: <output_dir>/<label_id>.json,
a JSON list of prompt strings.

Set the API key via the GOOGLE_API_KEY environment variable.
"""
import os
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from google import genai

from prompts import create_action_prompt, extract_json

# Roughly how many prompts one API call returns (the template asks for 25); used
# only to decide how many calls to launch per batch.
ESTIMATED_PROMPTS_PER_CALL = 25


def parse_args():
    p = argparse.ArgumentParser(description="Generate diverse per-class text-to-video prompts.")
    p.add_argument("--label_map", required=True,
                   help="data/annotations/<benchmark>/misc/label_id_to_label.json")
    p.add_argument("--action_info_dir", required=True,
                   help="directory of action profiles from action_profile_generation.py")
    p.add_argument("--output_dir", required=True,
                   help="where <label_id>.json prompt lists are written")
    p.add_argument("--num_prompts", type=int, default=330,
                   help="number of unique prompts per class (the per-dataset balance target)")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--seed", type=int, default=126)
    p.add_argument("--max_workers", type=int, default=16)
    return p.parse_args()


def generate_prompts_for_label(client, model, label, action_info):
    """One API call -> list of prompt strings (empty list on failure)."""
    try:
        prompt = create_action_prompt(label, action_info)
        response = client.models.generate_content(model=model, contents=prompt)
        return list(extract_json(response.text).values())
    except Exception as e:
        print(f"API call failed for '{label}': {e}")
        return []


def main():
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set the GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    label_id_2_label = json.load(open(args.label_map))

    for label_id, label in tqdm(label_id_2_label.items()):
        save_path = os.path.join(args.output_dir, f"{label_id}.json")
        if os.path.exists(save_path):
            print(f"label_id {label_id} already done, skipping.")
            continue

        profile_paths = [
            os.path.join(args.action_info_dir, f"label_{label_id}_gen_{i}.json")
            for i in range(5)
        ]
        profiles = [json.load(open(p)) for p in profile_paths if os.path.exists(p)]
        if not profiles:
            print(f"No action profiles for label_id {label_id}, skipping.")
            continue

        prompts = set()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            while len(prompts) < args.num_prompts:
                needed = args.num_prompts - len(prompts)
                num_calls = min(args.max_workers, (needed // ESTIMATED_PROMPTS_PER_CALL) + 1)
                futures = [
                    executor.submit(generate_prompts_for_label, client, args.model,
                                    label, random.choice(profiles))
                    for _ in range(num_calls)
                ]
                for future in futures:
                    prompts.update(future.result())

        with open(save_path, "w") as f:
            json.dump(list(prompts)[:args.num_prompts], f, indent=2)


if __name__ == "__main__":
    main()
