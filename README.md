# Constrained Distributional Counterfactual Explanations
The repository is the official implementation of the paper *Constrained Distributional Counterfactual Explanations*. It extends the original DCE (Distributional Counterfactual Explanation) algorithm with a constraint management framework to address the problem of generating distributional counterfactual explanations under feature constraints.

## 1. Core Modules
| File Path                | Core Function                                                                 |
|-------------------------|-------------------------------------------------------------------------------|
| `explainers/dce_v2.py`  | Implements the `DCEWithConstraints` class, extending the original DCE framework with constraint loss fusion and GPU adaptation |
| `explainers/manager.py` | Implements the `ConstraintManager` class, which unifies the scheduling of multiple constraint types including mean/std/LSC/FSD/SSD |
| `explainers/constraints/` | Contains concrete implementations of mean/std/LSC/FSD/SSD constraints, performing forward calculation and validation for each constraint type |
| `explainers/recovery/` | Post-experiment result recovery module:<br>• `RecoveryManager` unifies the scheduling of various recovery logics<br>• Includes recovery algorithms for metrics such as mean/std, and experiment result evaluators |

## 2. Experimental Resources
| Resource Path                  | Usage Description                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| `data/`                   | Stores raw experimental datasets                                                       |
| `HELOC/` / `cardio/` / `market/` | Independent experiment directories for each dataset:<br>• Contains experiment running scripts (e.g., `cardio_new.py`) responsible for transferring data/models to GPU<br>• Stores experiment result files (e.g., `.xlsx`) and running logs (`output_*.log`) |
| `scripts/run_validation_dataset.sh` | Batch script to run all experiments for the three validation datasets (HELOC / cardio / market) with one click |
| `HELOC_analysis.ipynb` / `cardio_analysis.ipynb` / `market_analysis.ipynb` | Dataset analysis Notebooks in the root directory, corresponding to the analysis of experimental results for the three validation datasets |
| `26_german_credit.ipynb` | Full-process file for German Credit case study:<br>• Includes experiment running and result analysis|
