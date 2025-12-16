# PSO–SVM Feature Selection (CENG482 Final Project)

This repository implements **feature selection using Particle Swarm Optimization (PSO) and Support Vector Machines (SVM)**, inspired by the paper *“Feature Selection using PSO-SVM”* (Chung‑Jui Tu et al.).

> Note: PDFs are ignored by git in this repo (`*.pdf` in `.gitignore`), so the paper/project description may exist locally under `Papers/` but won’t be committed.

### Project structure

- **`pso_svm_fs/`**: Core package
  - **`data_loading.py`**: Dataset loading helpers (WDBC, Wine, Sonar)
  - **`config.py`**: PSO + SVM hyperparameter dataclasses
  - **`svm_fitness.py`**: SVM-based fitness evaluation (cross-validation)
  - **`pso.py`**: Binary PSO implementation (feature mask search)
  - **`experiment.py`**: Single/multiple runs and JSON result saving under `results/<dataset>/`
- **`main_pso_svm.py`**: Example entrypoint (runs multiple experiments)
- **`Datasets/`**: Local dataset files (ignored by git)
- **`results/`**: Generated outputs (you may want to keep these tracked or ignore them depending on your workflow)

### Setup (recommended: virtual environment)

Create a local virtualenv and install dependencies:

```bash
./scripts/setup_venv.sh
```

Then activate it (each new terminal):

```bash
source .venv/bin/activate
```

### Run

```bash
python main_pso_svm.py
```

By default, it runs 30 repetitions for `dataset_name="wdbc"` and saves per-run JSON + a summary JSON under:

- `results/wdbc/`

### Datasets (expected locations)

The loaders are written assuming these paths exist locally:

- **WDBC**: `Datasets/breast+cancer+wisconsin+diagnostic/wdbc.data`
- **Wine**: `Datasets/wine/wine.data`
- **Sonar**: `Datasets/connectionist+bench+sonar+mines+vs+rocks/sonar.all-data`

### Notes

- Many functions are currently **stubs** (`raise NotImplementedError`)—fill them in incrementally while keeping the structure.
- If you want reproducible runs, keep `base_random_seed` in `run_multiple_experiments(...)`.


