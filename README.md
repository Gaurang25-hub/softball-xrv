# Softball xRV Modeling Pipeline

This repository contains a leakage-aware machine learning pipeline for modeling `xRV Of Event (Count Ind)` from TrackMan softball pitch-tracking data.

The project is organized as a full pre-evaluation workflow:

schema design → cleaning → grouped split → validation → train-only EDA → feature engineering → model selection.

At the current checkpoint, the pipeline is implemented through `modeling.py`. Final held-out evaluation on the untouched test set is the remaining step.

## Project Goal

The goal of this project is to build an interpretable, research-style xRV modeling pipeline from pitch-tracking data while protecting against data leakage and documenting each stage of the workflow.

This repository emphasizes:

- schema-driven data validation
- leakage-aware train/test splitting
- train-only exploratory data analysis
- physics-informed feature engineering
- disciplined model comparison across tree models and neural networks
- probability-aware and ordinal-aware evaluation metrics

## Current Project Status

Implemented:

- `config.py`
- `schema.py`
- `cleaning.py`
- `split.py`
- `validate.py`
- `plotting.py`
- `eda.py`
- `feature_sets.py`
- `metrics.py`
- `tree_models.py`
- `nn_train.py`
- `models/neural_net.py`
- `models/neural_net_losses.py`
- `modeling.py`

Pending:

- final outputs from `modeling.py`
- `evaluation.py`
- final held-out test-set results

## Repository Structure

```text
softball-xrv/
├── README.md
├── requirements.txt
├── .gitignore
├── .vscode/
│   └── launch.json
├── src/
│   └── softball_xrv/
│       ├── config.py
│       ├── schema.py
│       ├── cleaning.py
│       ├── split.py
│       ├── validate.py
│       ├── plotting.py
│       ├── eda.py
│       ├── feature_sets.py
│       ├── metrics.py
│       ├── tree_models.py
│       ├── nn_train.py
│       ├── modeling.py
│       └── models/
│           ├── neural_net.py
│           └── neural_net_losses.py
└── reports/
    ├── quality/
    ├── validation/
    ├── eda/
    └── model_selection/
```



## Feature Schema and Bounds

The Track A schema defines 17 pitch-level tracking features from release through front-of-home-plate crossing. Each feature includes a description, dtype, unit, hard bounds, soft bounds, and a missingness rule.

Soft bounds were estimated using Tukey’s IQR rule:

`lower = Q1 - 1.5 × IQR`, `upper = Q3 + 1.5 × IQR`.

These bounds represent the expected empirical operating range for each feature. Values inside the soft bounds are treated as typical pitch-level observations.

Hard bounds were set using domain and physics-based judgment. They represent broader sanity limits based on the inherent nature of each feature, such as nonnegative speed, realistic flight time, physically meaningful height values, and plausible release or plate-location ranges.

Values inside the hard bounds but outside the soft bounds are treated as suspicious but still usable. Values outside the hard bounds are treated as strong data-quality red flags.

Some features require special handling. `SpinAxis` is circular rather than linear, so soft bounds are omitted. `PlateLocHeight` was clamped at a lower bound of zero because pitch height cannot be physically negative.


## Data Splitting and Leakage Control

This project uses a leakage-aware train/test split rather than a random row split.

The cleaned interim dataset is first filtered to keep only rows with:

- a valid non-missing target value: `xRV Of Event (Count Ind)`
- a valid `GameID` value

The final train/test split is then created with `GroupShuffleSplit`, using `GameID` as the grouping column. This ensures that all pitches from the same game are assigned entirely to either train or test, preventing same-game leakage across the evaluation boundary.

`PitcherID` was considered as an ideal grouping variable because it could test whether models generalize across pitcher-specific throwing profiles. However, the available `PitcherID` column had too few usable unique IDs for a reliable grouped split, so `GameID` was used as the strongest available leakage-aware grouping fallback.

The held-out `test.csv` is not used for EDA, feature engineering decisions, model selection, or hyperparameter comparison. Those decisions are made using train-only analysis and an inner validation split. The test set is reserved for final evaluation only.

## Track A Validation Summary

After cleaning and splitting the data, `validate.py` checks the 17 Track A pitch-tracking features against the project schema. Each feature is evaluated for missingness, soft-bound coverage, hard-bound violations, and special handling rules.

All Track A features passed the 5% missingness threshold in both train and test. The maximum missingness was below 1%, which suggests that numeric coercion did not create a large amount of unusable feature data.

The soft bounds were based on Tukey-IQR empirical ranges. Most pitch-level observations fell inside these soft intervals, indicating that the feature distributions are largely concentrated within expected operating ranges rather than being dominated by extreme tails.

The hard bounds were domain/physics-informed sanity limits. In both train and test, all features except `PlateLocHeight` had 0% outside-hard-bound values. `PlateLocHeight` had a small number of outside-hard-bound observations, about 1.22% in train and 0.82% in test, so it was flagged for extra awareness but not removed automatically.

This validation step supports the quality of the Track A baseline feature set. It is used as a data-quality audit, not for model selection. The held-out test set remains reserved for final evaluation and is not used to make feature-engineering or model-selection decisions.


## Validation-Informed Distribution Inspection

The Track A validation report showed that most baseline pitch-tracking features had low missingness and no hard-bound violations. The only feature with nonzero outside-hard-bound values was `PlateLocHeight`, with about 1.22% outside the hard interval in train and 0.82% in test.

Because `PlateLocHeight` is also used to construct derived vertical-trajectory features such as `DeltaZ` and `DeltaZ_over_ZoneTime`, I inspected these variables more closely using histograms, KDE curves, boxplots, and percentile summaries.

The inspection showed that `PlateLocHeight` was still concentrated around a realistic central range, with only a small tail of extreme values. `DeltaZ` and `DeltaZ_over_ZoneTime` also showed coherent unimodal distributions rather than signs of widespread corruption.

Based on this audit, `PlateLocHeight` was retained, but flagged as a feature requiring extra awareness. This step connects the schema validation stage to the EDA stage: validation identified the feature requiring special attention, and EDA confirmed that the feature and its derived variants were still usable.


## Validation-Informed EDA

The validation report identified `PlateLocHeight` as the only Track A feature with a small number of hard-bound violations in the training split. Because `PlateLocHeight` is also used to construct vertical-trajectory features such as `DeltaZ` and `DeltaZ_over_ZoneTime`, these variables were inspected with train-only histograms, KDE curves, boxplots, and summary statistics.

The distributions were approximately unimodal and concentrated around realistic central ranges, with limited tail behavior rather than widespread corruption. Based on this train-only inspection, `PlateLocHeight` and its derived vertical-trajectory features were retained, but documented as features requiring extra awareness.

This step connects validation to EDA: schema checks identified the feature needing special attention, and train-only distribution inspection supported retaining it for downstream modeling.



