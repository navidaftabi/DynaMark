# DynaMark

**DynaMark** is a reinforcement-learning framework for **dynamic watermarking (DWM)** to detect **replay attacks** in industrial control systems (with emphasis on industrial machine tool controllers). It learns an **adaptive watermark-covariance policy** that balances detection belief, actuation energy, and control performance.

<p align="center">
  <img src="img/fw.png" alt="DynaMark framework" width="900">
</p>

**Paper:** `[PAPER_HERE]` 
- Link: `[LINK_HERE]`

---

## Highlights
- **Adaptive (non-stationary) watermarking:** learns to dynamically adapt watermark intensity over time based on system dynamics rather than using a constant-variance watermark.
- **Detector-aware state:** uses residual-based test statistics belief/confidence signal to guide watermark adaptation.
- **Replay-attack focus:** designed to detect measurement replay while an attacker manipulates the control/plant behavior.
- **Multiple case studies / environments:** digital-twin and simulation benchmarks included in this repo.

---

## Repository structure
- `config/` — experiment configs (YAML)  
  - `dt/` (digital twin), `msd/` (nonlinear MSD), stepper motor (digital twin) `sm/continuous/`, `sm/discrete/`
- `env/` — environment + plant models + detector/belief logic  
  - `env/plants/`: `dt_linear.py`, `msd_nonlinear.py`, `sm_dt_continuous.py`, `sm_dt_discrete.py`
- `src/` — training, evaluation, policies, baselines, plotting utilities  
  - `src/train/` (DDPG training, checkpointing, logging)  
  - `src/eval/` (rollouts, evaluation pipeline, policies, I/O)  
  - `src/baseline/` (baseline watermarking routines)  
  - `src/policies/` (policy networks + constant policies)  
  - `src/plots/` (timeseries + curves utilities)
- `data/` — input data files used by experiments
- `output/` — generated outputs (checkpoints, JSONL logs, rollouts)
- `img/` — figures 

---

## Installation

This repository uses Conda. Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate dwm
```

---

## Quickstart

### Train a DynaMark policy
Training configs are under `config/*/train.yaml` (or case-specific equivalents).

```bash
# Digital twin (DT)
python -m src.train.train_ddpg --config config/dt/train.yaml

# Nonlinear MSD
python -m src.train.train_ddpg --config config/msd/train.yaml

# Stepper-motor discrete DT
python -m src.train.train_ddpg --config config/sm/discrete/train.yaml
```

Training artifacts are saved under `output/<case>/train/` by default (e.g., `ckpt_best.pt`, `episodes.jsonl`, `learning_curve.png`). Exact paths are defined in the YAML configs.

---

### Evaluate a trained policy

```bash
# Digital twin (DT)
python -m src.eval.run --config config/dt/evaluate.yaml

# Nonlinear MSD
python -m src.eval.run --config config/msd/evaluate.yaml

# Stepper-motor discrete DT
python -m src.eval.run --config config/sm/discrete/evaluate.yaml
```

Evaluation results are written under `output/<case>/evaluate/`, typically including rollouts for:
- `normal/`
- `attack_replay_on*/` (replay attacks with different onset times)

---

### Baselines
#### Benchmark:
 Baseline routines located in `src/baseline/` is the implementation of [An online approach to physical watermark design](https://ieeexplore.ieee.org/abstract/document/9061046). This benchmark contains optimal watermarking design under linear system dynamics assumption:
- `Offline`: With known system dynamics parameters, solves an optimization problem
- `Online`: With unknown system dynamics parameters, system identification + optimal watermark design

```bash
# Example for online version: stepper-motor continuous baseline config 
python -m src.baseline.run --config config/sm/continuous/tac.yaml
```
#### Constant-Covariance Baselines:
The routin in `src/eval` can simulate any of the case-studies under constant-covariance baselines. For this baseline, please see the `evaluate.yaml` under `configs/` for each case-study. To run this baseline follow the steps for [policy evaluation](###evaluate-a-trained-policy) after re-configuring the `evaluate.yaml` file.


---

## Outputs & logging

Common outputs include:
- `output/<case>/train/ckpt_best.pt`, `ckpt_latest.pt`
- `output/<case>/train/episodes.jsonl`, `eval.jsonl`
- `output/<case>/train/learning_curve.json` (+ `.png`)
- evaluation rollouts under `output/<case>/evaluate/...`

Plot utilities in `src/plots/` can be used to generate time-series figures from the stored JSONL and rollout files.

---

## Reproducing paper results
1. Create the Conda environment (`environment.yml`).
2. Train/evaluate each case study via its YAML configs in `config/`.
3. Generate figures using plotting workflow (`src/plots/` or your scripts), writing outputs to `img/`.

---

## Citation
If you use this code, please cite the arXiv preprint:

```bibtex
@misc{dynamark,
  title   = {DynaMark: A Reinforcement Learning Framework for Dynamic Watermarking in Industrial Machine Tool Controllers},
  author  = {Author(s)},
  note    = {Note},
  year    = {2026},
  url     = {LINK_HERE}
}
```

---

## Contact
For questions/issues: open a GitHub Issue or contact the authors.
