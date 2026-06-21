# Gen2Balance video generation pipeline

Code for generating synthetic videos for long-tailed benchmarks. Two stages: generate
diverse text prompts per class (Gemini), then generate videos from those prompts
(Wan 2.1 text-to-video). Output videos are named `<label_id>_<sample_idx>.mp4`,
matching the file names used in our project (`data/annotations/<benchmark>/balanced/train.csv`).

```
generation/
├── text_prompt_generation/
│   ├── prompts.py                      # Gemini prompt templates
│   ├── action_profile_generation.py    # (1) generate per-class "action profiles" using exemplar videos
│   └── text_prompt_generation.py       # (2) generate per-class diverse text prompts  -> <label_id>.json
└── video_generation/
    ├── build_manifest.py               # (3) concatenate per-class prompts -> one manifest
    ├── generate_videos.py              # (4) Wan 2.1 T2V -> <label_id>_<sample_idx>.mp4
    ├── generate.sh                     # single-job wrapper
    └── slurm_run_generation.sh         # SLURM job array wrapper
```

## Requirements

- **Text prompt generation**: `pip install google-genai tqdm`, and
  `export GOOGLE_API_KEY=...` (the scripts read the key from the environment).
- **Video generation**: `pip install diffusers transformers accelerate imageio[ffmpeg]`.
  The Wan 2.1 weights (`Wan-AI/Wan2.1-T2V-14B-Diffusers`) are auto-downloaded from
  the Hugging Face Hub on first run; pass a local path via `--model_path` to use a
  cached copy.

## Usage

Set `BENCH` to one of `K100-LT`, `UCF-LT`, `RareAct-K100-LT`, and `N` to the
per-class balance target (e.g., 990 for K100-LT). `DATA_ROOT` is your video root 
which contains real videos of `BENCH` (see the top-level README).

**1. Action profiles**: sample 5 real exemplar videos per class and have Gemini
deconstruct each action. Exemplars are read straight from the real `train.csv`.

```bash
python text_prompt_generation/action_profile_generation.py \
    --label_map  ../data/annotations/$BENCH/misc/label_id_to_label.json \
    --train_csv  ../data/annotations/$BENCH/real/train.csv \
    --data_root  $DATA_ROOT \
    --output_dir profiles/$BENCH
```

**2. Text prompts**: turn the profiles into `N` unique prompts per class
(`profiles/$BENCH/<label_id>_*` → `prompts/$BENCH/<label_id>.json`).

```bash
python text_prompt_generation/text_prompt_generation.py \
    --label_map       ../data/annotations/$BENCH/misc/label_id_to_label.json \
    --action_info_dir profiles/$BENCH \
    --output_dir      prompts/$BENCH \
    --num_prompts     $N
```

**3. Build the manifest**: concatenate the per-class prompt files into a single
ordered manifest.

```bash
python video_generation/build_manifest.py \
    --prompts_dir    prompts/$BENCH \
    --output         video_prompts.json \
    --mapping_output prompts.json
```

The manifest is a flat list `[{"name": "0_0", "label_id": 0, "sample_idx": 0,
"prompt": "..."}, ...]` so array jobs slice it by global index, while each clip
is saved under its final name.

**4. Generate videos**: run Wan 2.1 over the manifest. Each entry is saved as
`<name>.mp4`, so there is no renaming step. Edit the paths at the top of
`generate.sh`, then either run a slice directly or submit the SLURM array:

```bash
# one local slice (indices 0..56 of the manifest)
bash video_generation/generate.sh 0 56

# or the full set as a SLURM array (set --array to ceil(len(manifest)/VIDEOS_PER_JOB)-1)
sbatch video_generation/slurm_run_generation.sh
```

`generate_videos.py` skips clips that already exist, so array jobs are
restartable.

## Notes

- The `balanced` splits fill each class **up to** `N` (so generated counts vary
  per class; head classes need few or none). Generate `N` prompts per class and
  the `balanced` CSVs reference exactly the subset they need.
- For `RareAct-K100-LT`, only the rare-action classes (label ids 100–121) need
  new clips; the Kinetics-100 classes reuse the K100-LT generated clips.

## License & citation

Videos are generated with [Wan 2.1](https://github.com/Wan-Video/Wan2.1); its
license is included as `WAN_LICENSE.txt` in the released data. Please cite **Wan**
(and **VideoMAE**) alongside Gen2Balance if you use this pipeline.
