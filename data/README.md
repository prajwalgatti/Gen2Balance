# Data

This folder holds the **annotations** (train/test splits for the three
benchmarks) and explains how to assemble the **videos** they reference. The repo
does not contain the videos: the *real* clips come from the original datasets
(Kinetics-400, UCF-101, RareAct), and the *generated* clips + model checkpoints are
released separately (download from the [project page](https://prajwalgatti.github.io/gen2balance)).

`data/` is also the **video root** (or `DATA_ROOT`), so once populated it should look like:

```
data/                              # = DATA_ROOT
├── README.md
├── verify_videos.py
├── annotations/                   # the split/label files
│   └── K100-LT/  UCF-LT/  RareAct-K100-LT/
├── Kinetics-400/                  # you provide
│   ├── videos_train/  videos_val/  K100_generated_videos/
├── UCF/                           # you provide
│   ├── videos_mp4/  UCF_generated_videos/
└── RareAct/                       # you provide
    ├── train/  test/  RareAct_generated_videos/
```

You do **not** have to figure out which clips to use. Every split in `annotations`
lists the exact files it needs, as paths relative to `DATA_ROOT`, e.g.:

```
Kinetics-400/videos_train/6rcGJOvsCA4.mp4 60
UCF/videos_mp4/v_ApplyEyeMakeup_g11_c01.mp4 0
Kinetics-400/K100_generated_videos/1_0.mp4 1
```

So the task is: obtain each dataset and place its videos in this dir.
All filenames are preserved from the original datasets.

---

## Annotation format

| Benchmark          | `--data_set`     | Classes | Source                           |
|--------------------|------------------|---------|----------------------------------|
| `K100-LT`          | `Kinetics-100`   | 100     | 100-class subset of Kinetics-400 |
| `UCF-LT`           | `UCF101`         | 101     | UCF-101                          |
| `RareAct-K100-LT`  | `Kinetics-100-RA`| 122     | K100 + rare actions from RareAct |

Each benchmark folder has the same structure (K100-LT additionally ships a
`balanced_330` scaling variant):

```
annotations/<benchmark>/
├── balanced/        # for STAGE 1 of training: real + generated clips, fully class-balanced
│   ├── train.csv
│   ├── test.csv
│   └── train_freq.json
├── real/            # for STAGE 2 of training: real long-tailed clips only
│   ├── train.csv
│   ├── test.csv
│   └── train_freq.json
├── balanced_330/    # (K100-LT only) partial-balancing variant used in ablations (B=330 clips/class)
└── misc/
    ├── label_id_to_label.json
    └── lt_mapping.json
```

**`train.csv` / `test.csv`** has one clip per line: `<relative_video_path> <label_id>`.
**`train_freq.json`** has per-class training-clip counts ordered by
`label_id` (for balanced softmax loss). **`misc/label_id_to_label.json`** has `label_id`
to class name mapping. **`misc/lt_mapping.json`** has class name to long-tail group
(`head` / `tail` / `fewshot`, plus `rareact` for RareAct-K100-LT) mapping. 

The number of classes per long-tail groups are:

| Benchmark         | head | tail | fewshot | rareact |
|-------------------|------|------|---------|---------|
| K100-LT           | 11   | 59   | 30      | -       |
| UCF-LT            | 18   | 28   | 55      | -       |
| RareAct-K100-LT   | 11   | 59   | 30      | 22      |

---

## Getting the videos

### Real clips (download from the original datasets)

| Benchmark | Dataset | Source | Place under |
|-----------|---------|--------|-------------|
| K100-LT / RareAct-K100-LT (K100 classes) | Kinetics-400 (pre-processed mp4) | https://opendatalab.com/Kinetics-400 | `Kinetics-400/videos_train/`, `Kinetics-400/videos_val/` |
| UCF-LT | UCF-101 | https://www.crcv.ucf.edu/data/UCF101.php | `UCF/videos_mp4/` |
| RareAct-K100-LT (rare-action classes) | RareAct (curated clips) | `RareAct.tar` from the release (below) | `RareAct/train/`, `RareAct/test/` |

> UCF-101 ships `.avi`; the CSVs expect `.mp4` under `UCF/videos_mp4/`. Convert the
> clips to mp4 keeping the original basenames.

### Generated clips (released by us)

All generated clips, the prompts, and the trained checkpoints
are available for download on the [project page](https://prajwalgatti.github.io/gen2balance/).
The download folder structure is laid out as:

```
.
├── dataset/
│   ├── README.txt
│   ├── WAN_LICENSE.txt
│   ├── RareAct.tar             # real RareAct clips (train + test)
│   ├── K100LT_generated.tar    # generated K100LT clips
│   ├── UCFLT_generated.tar     # generated UCFLT clips
│   ├── RareAct_generated.tar   # generated RareAct clips
│   └── prompts.tar             # text prompt corresponding to each generated video
└── checkpoints/                # trained Gen2Balance models (see top-level README)
    ├── K100-LT/  K100-LT-partial-balanced/  UCF-LT/  RareAct-K100-LT/
    └── README.txt
```

Extract the released tarballs into `DATA_ROOT`:

| Tarball | Extract to |
|---------|------------|
| `K100LT_generated.tar`   | `Kinetics-400/K100_generated_videos/` |
| `UCFLT_generated.tar`    | `UCF/UCF_generated_videos/` |
| `RareAct_generated.tar`  | `RareAct/RareAct_generated_videos/` |

If you'd like to regenerate them from scratch instead, see [`../generation/`](../generation).
The generated clips are produced with Wan 2.1. See `WAN_LICENSE.txt` in the
release and please cite WAN.

---

## Verify your setup

Once `DATA_ROOT` is populated, check every referenced clip exists before a run:

```bash
python data/verify_videos.py --data_root data --benchmark K100-LT
```

It scans the benchmark's CSVs and reports any missing files.
