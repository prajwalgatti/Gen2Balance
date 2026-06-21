"""Generate videos from a prompt manifest with Wan 2.1 (T2V).

Reads the manifest produced by build_manifest.py (a flat list of
{"name", "label_id", "sample_idx", "prompt"}), and writes one mp4 per entry
named "<name>.mp4" (i.e. "<label_id>_<sample_idx>.mp4") directly into the output
directory — no global-index naming and no post-hoc renaming.

For parallelism across a SLURM array, slice the manifest with --start_idx /
--end_idx; each job handles its own contiguous range of the global list.
"""
import os
import json
import argparse
import concurrent.futures

import torch
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-14B-Diffusers" 


def export_video_imageio(frames, output_path, fps=16):
    """Encode a list of PIL Images to an mp4 with imageio/ffmpeg."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames_array = [np.array(frame.convert("RGB")) for frame in frames]
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
        ffmpeg_params=["-preset", "ultrafast", "-profile:v", "baseline", "-level", "3.0"],
    )
    try:
        for frame in frames_array:
            writer.append_data(frame)
    finally:
        writer.close()
    return output_path


def process_and_save_video(video_frames, save_path, fps=16):
    video = (video_frames * 255).astype(np.uint8)
    video = [Image.fromarray(frame) for frame in video]
    export_video_imageio(video, save_path, fps=fps)


def generate(pipe, entries, batch_size=6, num_frames=33, height=480, width=832,
             guidance_scale=5.0, num_inference_steps=50, save_root_dir="outputs"):
    os.makedirs(save_root_dir, exist_ok=True)

    for batch_start in tqdm(range(0, len(entries), batch_size), desc="Batches"):
        batch = entries[batch_start: batch_start + batch_size]
        text_prompts = [e["prompt"] for e in batch]
        names = [e["name"] for e in batch]

        videos = pipe(
            prompt=text_prompts,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), 8)) as executor:
            futures = [
                executor.submit(process_and_save_video, videos[i],
                                os.path.join(save_root_dir, f"{names[i]}.mp4"), 16)
                for i in range(len(batch))
            ]
            concurrent.futures.wait(futures)

        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Generate videos from a prompt manifest with Wan 2.1.")
    parser.add_argument("--prompts_file", required=True,
                        help="manifest from build_manifest.py (list of {name, prompt, ...})")
    parser.add_argument("--output_dir", default="./outputs/", help="where <name>.mp4 files are written")
    parser.add_argument("--start_idx", type=int, default=0, help="start index into the manifest (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None, help="end index into the manifest (exclusive); default = end")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_path", default=DEFAULT_MODEL,
                        help="HF model id (auto-download) or local path to Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--seconds", type=int, default=4, help="video length in seconds")
    args = parser.parse_args()

    manifest = json.load(open(args.prompts_file))
    end_idx = len(manifest) if args.end_idx is None else min(args.end_idx, len(manifest))
    entries = manifest[args.start_idx:end_idx]

    # Resume: skip entries already generated.
    entries = [e for e in entries
               if not os.path.exists(os.path.join(args.output_dir, f"{e['name']}.mp4"))]
    print(f"Generating {len(entries)} videos (manifest range {args.start_idx}:{end_idx}) -> {args.output_dir}")
    if not entries:
        print("Nothing to do.")
        return

    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.bfloat16)
    pipe = WanPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    pipe.to("cuda")
    print("Model loaded.")

    num_frames = 16 * args.seconds + 1
    generate(
        pipe,
        entries,
        batch_size=args.batch_size,
        num_frames=num_frames,
        height=480,
        width=832,
        guidance_scale=5.0,
        num_inference_steps=50,
        save_root_dir=args.output_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
