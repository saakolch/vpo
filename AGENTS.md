# AGENTS.md

You are assisting with an AI/ML research codebase targeting NeurIPS/ICLR/ICML.

Core rules:
1. Never change the evaluation protocol without explicit approval.
2. Never change train/val/test splits without explicit approval.
3. Never report a metric unless it is produced by scripts/eval.py.
4. Never delete experiment logs.
5. Always preserve random seeds and config files.
6. Every code change must be accompanied by:
   - changed files
   - reason for change
   - tests run
   - expected effect
7. For any bug fix, add or update a test if possible.
8. For any new experiment, create a config under configs/.
9. For any result table, include:
   - mean
   - std
   - number of seeds
   - dataset
   - metric
   - checkpoint path
   - command used
10. Do not write paper claims that are not supported by experiments.

Project-specific operating rules:
1. Read this file and `pre_experiments/STATUS.md` first.
2. Hydra is the only experiment-configuration system for scientific parameters.
3. Never hardcode scientific settings in scripts.
4. Never store secrets or W&B API keys in repo files, configs, logs, manifests, or W&B config.
5. All current outputs belong under `pre_experiments/`.
6. Scripts must refuse to write outside `pre_experiments/` unless `safety.allow_main_experiments=true` or `--allow_main_experiments=true` is explicit.
7. UltraFeedback is the primary scientific benchmark for the V3 protocol; EUREQA is secondary, MuSiQue is tertiary/optional, and Maze is legacy/debug-only.
8. Run benchmark maturity in order: parser/reward tests, frozen base-model sampling/evaluation, then training integration.
9. Update the source ledger and status after changes.
10. Run tests before GPU jobs.
11. No job above 4 GPU-hours without explicit authorization.
12. Preserve failed runs and attempt IDs.
13. Do not claim exact reproduction while assumptions remain.
14. Finish each work session with exactly one `pre_experiments/NEXT_ACTION.md` recommendation.

Before editing code:
- summarize the current implementation
- identify the minimal files to change
- propose a test plan

After editing code:
- run unit tests if available
- run a smoke test
- summarize the diff

## Training

These GPU notes are HPC2 site notes, not portable release instructions. Keep
scientific settings in Hydra experiment configs, and keep cluster-specific
settings such as partition, GPU type, wall time, cache root, and environment
paths in site configs or launch-time overrides. A800 is the default choice on
HPC2 for serious runs, but it must not be hardcoded as a scientific
requirement.

HPC2 GPU partitions currently available:

Partition	GPUs	Nodes	Timelimit	Notes
i64m1tga800u	A800 x8/node	gpu1-[2,15,20,25,31,35,39,41,48]	7 days	General
i64m1tga800ue	A800 x8/node	same	7 days	Higher priority
i64m1tga40u	A40 x8/node	gpu3-[1,6,13]	7 days	General
i64m1tga40ue	A40 x8/node	same	7 days	Higher priority
long_gpu	A800 x8/node	same as 800u	14 days	Long jobs
emergency_gpu	A800 x8/node	same as 800u	14 days	Emergency
debug	A40	gpu3-9	30 min	Quick tests

srun examples for HPC2 interactive debugging:

```bash
# 1 A800 GPU, interactive shell
srun -p i64m1tga800u --gres=gpu:a800:1 --pty bash

# 2 A800 GPUs
srun -p i64m1tga800u --gres=gpu:a800:2 --pty bash

# 1 A40 GPU
srun -p i64m1tga40u --gres=gpu:a40:1 --pty bash

# Quick debug test (30 min max)
srun -p debug --gres=gpu:a40:1 --pty bash
```

sbatch example for HPC2:

```bash
#!/bin/bash
#SBATCH -p i64m1tga800u          # partition
#SBATCH --gres=gpu:a800:1        # 1 A800 GPU
#SBATCH -c 8                     # CPU cores
#SBATCH --mem=64G                # memory
#SBATCH -t 2-00:00:00            # 2 days
#SBATCH -o slurm_%j.out          # stdout
#SBATCH -e slurm_%j.err          # stderr

# Load your environment (ONLY IF NECESSARY FOR GPU RUN)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Run your training
python train.py --config configs/my_config.yaml
```

Then submit:

```bash
sbatch run.sh
```

Useful commands:

```bash
squeue -u $USER          # check your jobs
scancel <job_id>         # cancel a job
sinfo -p i64m1tga800u   # check partition availability
```

Use `i64m1tga800u` or `i64m1tga800ue` for HPC2 serious training when available.
Use the `debug` partition only for short smoke tests. Always check live
availability with `sinfo` or `squeue`; partition state is point-in-time.

## V3 protocol

The active protocol is `VPO_base_diagnostics_and_training_master_prompt_v3.md`.
The previous Maze-primary protocol is deprecated but preserved for audit history.
Maze may be used for parser/debug or veRL integration smoke tests only; it must not
gate scientific progress or be described as the primary benchmark.

# vLLM, veRL, and reproducibility

CUDA is not visible on login nodes. GPU import checks, vLLM generation, and veRL
training must run on a GPU node through `srun` or `sbatch`.

HPC2 local paths:

- `~/tools/envs/vllm/bin/python`: standalone vLLM environment for inference
  smokes and cached-model generation tests. This is a local HPC2 path, not a
  portable release dependency.
- `/hpc2hdd/home/skolchin784/envs/verl_vpo/bin/python`: local veRL/VPO training
  environment. This is also a local HPC2 path, not a portable release
  dependency.
- `/hpc2hdd/home/skolchin784/tools/verl_src`: local veRL source checkout used
  by the current training environment.

HPC2 cache environment:

```bash
export HF_HOME=~/tools/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
```

Do not assume the same cache layout on another machine. Do not commit cached
models. Portable configs should record model IDs, exact revisions, tokenizer
revisions, chat-template hashes, reward-model revisions, and dataset split
hashes. Local cache roots and snapshot paths belong in site configs, resolved
configs, or run manifests.

For release reproducibility (later):

- Provide a small CPU/dev environment for tests and diagnostics.
- Provide a separate pinned GPU environment for veRL/vLLM training.
- Treat `pip freeze` as provenance, not as the primary install recipe.
- Prefer `pip install -e .` over relying on manual `PYTHONPATH=src`.
- Keep Docker or Apptainer/Singularity recipes as the stronger optional
  reproducibility path for users whose systems support containers.
- Put machine-specific paths under `configs/site/` or environment variables
  such as `VPO_VLLM_PYTHON`, `VPO_VERL_PYTHON`, `VPO_VERL_SOURCE_ROOT`,
  `HF_HOME`, and `RAY_TMPDIR`.
- Never put W&B API keys, Hugging Face tokens, or other secrets in repo files,
  configs, logs, manifests, or W&B config.
