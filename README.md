# Gen2Balance: Generative Balancing for Long-Tailed Video Action Recognition

### [[Project Website :dart:]](https://prajwalgatti.github.io/gen2balance/)&nbsp;&nbsp;&nbsp;[[Paper :page_with_curl:]](https://github.com/prajwalgatti/Gen2Balance)&nbsp;&nbsp;&nbsp;[Code :octocat:]

Official code for the ECCV'26 paper [Gen2Balance: Generative
Balancing for Long-Tailed Video Action Recognition](https://github.com/prajwalgatti/Gen2Balance).

---

## Installation

Reference setup: Python 3.10 + PyTorch 2.5.1 (CUDA) + torchvision 0.20.1.

```bash
conda create -n gen2bal python=3.10 -y
conda activate gen2bal

# CUDA-matched build (adjust the CUDA build for your system)
conda install -c conda-forge "pytorch=2.5.1=*cu*"
conda install --no-deps -c conda-forge torchvision=0.20.1

# remaining deps
pip install -r requirements.txt
```

Notes:
- **decord**: any recent version works (`pip install decord`, or build from source
  if there is no wheel for your platform).
- **timm**: pinned to `0.4.12` in `requirements.txt`. The code imports
  `timm.models.registry` / `timm.models.layers`, which were removed/relocated in
  `timm >= 0.9`, so do not bump this.

---

## Pretrained backbone

Gen2Balance fine-tunes from a pretrained VideoMAE ViT-B checkpoint:

```bash
cd checkpoints
bash download_videomae_checkpoint.sh   # -> saves checkpoints/mae_pretrain_vit_base.pth
```

### Trained Gen2Balance checkpoints

The trained stage-1/stage-2 models for each benchmark (K100-LT, UCF-LT, and RareAct-K100-LT) are released (download from
the [project page](https://prajwalgatti.github.io/gen2balance/)).
In the training/eval scripts, set the eval `MODEL_PATH` to
`checkpoint-stage1.pth` or `checkpoint-stage2.pth` for the benchmark you're interested in.

---

## Dataset setup

The annotations in [`data/annotations/`](data/annotations) are the exact train/test splits used in each benchmark; you assemble the videos under `data/` (the video root). 

See [`data/README.md`](data/README.md) for full instructions, sources to download datasets from, and released generated videos for each benchmark.

---

## Generation pipeline

The `balanced` splits mix real clips with generated videos produced by our pipeline. We **release all
generated videos** (see [`data/README.md`](data/README.md)), so reproducing our
results does **not** require running the pipeline.

To generate your own, the pipeline in [`generation/`](generation) (1) writes
diverse, class-faithful prompts with **Gemini 2.5 Pro** (requiring a `GOOGLE_API_KEY`)
then (2) synthesizes videos with **Wan 2.1**. See
[`generation/README.md`](generation/README.md) for the full workflow. Note: an
open-source MLLM such as [Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) can also be used
instead of Gemini; we found both models to perform comparably.

---

## Training & evaluation

Gen2Balance uses a two-stage recipe per benchmark:

- **Stage 1**: train on the `balanced` split (real + generated, class-balanced),
  initialized from the pre-trained VideoMAE backbone, with balanced softmax loss.
- **Stage 2**: continue training on the `real` (long-tailed) split, initialized
  from the stage-1 checkpoint.
- **Eval**: evaluate the final stage-2 checkpoint on the test split.

Stage 1 saves `checkpoint-stage1.pth` and stage 2 saves
`checkpoint-stage2.pth` (via `--ckpt_name`): stage 2 initializes from
`stage1/checkpoint-stage1.pth`, and eval loads `stage2/checkpoint-stage2.pth`.

Training and testing scripts are stored in `scripts/<benchmark>/`. Edit `DATA_ROOT` (and, if needed,
`OUTPUT_DIR`/`MODEL_PATH`) at the top of each, then submit in order:

```bash
# Example: K100-LT
sbatch scripts/K100-LT/g2b_K100LT_stage1.sh   # saves to output/K100LT/stage1/
sbatch scripts/K100-LT/g2b_K100LT_stage2.sh   # uses stage-1 checkpoint; saves to output/K100LT/stage2/
sbatch scripts/K100-LT/g2b_K100LT_eval.sh     # evaluates stage-2 checkpoint
```

The same three-step procedure applies to `UCF-LT/` and `RareAct-K100-LT/`. The scripts target a single-GPU SLURM setup; adjust `#SBATCH` directives,
`--nproc_per_node`, and `--batch_size` for your cluster.

**Without SLURM**, run any script directly with `bash` instead of `sbatch` (e.g.,
`bash scripts/K100-LT/g2b_K100LT_stage1.sh`).

The eval step reports the long-tailed metrics:
overall class-average accuracy along with per-category accuracy for the
head / tail / fewshot (and `rareact`, for RareAct-K100-LT) splits. For example:

```
Class-avg acc: 58.7%
  head:    74.2%
  tail:    55.1%
  fewshot: 41.8%
```

---

## Acknowledgements

This code is built on top of [VideoMAE](https://github.com/MCG-NJU/VideoMAE).
The original VideoMAE license and attribution are retained in `LICENSE` and
`NOTICE.md`. The generation pipeline uses [Wan 2.1](https://github.com/Wan-Video/Wan2.1) for
text-to-video synthesis. Please cite VideoMAE and Wan in addition to Gen2Balance if you use this codebase.

## License

This code is derivative of [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and is released under
the same **CC BY-NC 4.0** terms; see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
For **non-commercial research use only**. The generation pipeline and released artifacts
carry their own upstream terms: Wan 2.1 ships `WAN_LICENSE.txt` with the released data, and the Gemini API is a
hosted service.

## Citation

```bibtex
@inproceedings{gatti2026gen2balance,
  title     = {Gen2Balance: Generative Balancing for Long-Tailed Video Action Recognition},
  author    = {Gatti, Prajwal and Jenni, Simon and Caba Heilbron, Fabian and Damen, Dima},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

