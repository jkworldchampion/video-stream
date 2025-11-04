# Repository Guidelines

## Project Structure & Module Organization
- `video_depth_anything/`: Teacher (`video_depth.py`) and streaming student (`video_depth_stream.py`) backbones plus KD utilities under `aux/`.
- `data/`, `dataset/`: Dataset lists, loaders, and transforms shared by training and validation scripts.
- `utils/`: Losses (`loss_kd_aux.py`, `loss_MiDas.py`), training helpers, and I/O utilities reused across entrypoints.
- `loss/`, `metric_depth/`: Metric-specific criteria and evaluation helpers invoked during validation.
- `benchmark/`, `outputs*/`, `logs/`: Validation configs, rendered depth maps, and experiment artefacts—keep heavy outputs out of Git.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create an isolated environment aligned with your CUDA/PyTorch toolchain.
- `pip install -r requirements.txt`: Install runtime and logging dependencies.
- `python train.py --pretrained_ckpt checkpoints/video_depth_anything_vits.pth`: Launch KD training; checkpoints/metrics land in `checkpoints/` and `logs/`.
- `python train.py --test --resume_from <ckpt>`: Validation-only sweep on KITTI/ScanNet mini splits to confirm deltas.
- `python run_streaming.py --input_video assets/example_videos/davis_rollercoaster.mp4 --use_causal_mask`: Streaming sanity check; writes depth video to `outputs/`.
- `python run.py --input_video <path>`: Baseline non-streaming depth export for comparison.

## Coding Style & Naming Conventions
- Use 4-space indentation, snake_case functions, and PascalCase classes; match existing tensor names (`feat_s`, `qkv_aux`) for clarity.
- Keep docstrings and inline shape notes current when you touch forward passes or config knobs; favour concise comments over verbose prose.
- Only run formatters (`black`, `ruff`) on files you touch and avoid formatting-only commits.

## Testing Guidelines
- Quick regression: `python train.py --test --val_scenes 2` (≈10 min) logs delta1/AbsRel to `logs/`.
- Place ad-hoc repro scripts under `benchmark/` with `test_*.py` naming and ensure they run headlessly.
- Surface validation metrics plus representative frames from `outputs/` or `outputs_streaming/` in review threads.

## Commit & Pull Request Guidelines
- Match history with short imperative summaries (e.g., `clip based inference`); skip trailing periods.
- Scope commits narrowly—separate data updates, model tweaks, and script changes to keep diffs reviewable.
- PR descriptions must cover motivation, configs touched, validation evidence, and a checklist for data/checkpoint dependencies before review.

## Security & Configuration Tips
- Keep credentials (e.g., W&B keys) in a local `.env`; `train.py` loads them via `dotenv`.
- Use `get_weights.sh` to fetch checkpoints but reference filenames/config paths instead of tracking binaries; keep dataset roots configurable via CLI or YAML.
