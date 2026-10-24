# Beyond the 40-Yard Dash: What Combine Route Movement Reveals About NFL Receiving

Research project for the **NFL Big Data Bowl 2027** exploring whether whole-route movement patterns from NFL Combine tracking data contain information about future NFL receiving performance beyond conventional athletic measurements.

## Overview

The 40-yard dash, vertical jump, and other Combine measurements are widely used to evaluate athletic potential. However, these measurements reduce movement ability to a small number of isolated tests.

This project asks a different question:

> **Does the way a receiver moves through an entire route at the Combine reveal something about how efficiently they will perform in the NFL?**

Using approximately 10 Hz Combine tracking data, this project extracts whole-route movement features from receiver drills and tests whether these features provide incremental predictive information about future regular-season NFL receiving efficiency.

The analysis focuses particularly on **speed variability across a complete route**, rather than simply measuring maximum speed.

---

## Research Question

The central research question is:

**Can whole-route Combine movement features improve prediction of future NFL receiving efficiency beyond conventional Combine measurements such as the 40-yard dash and vertical jump?**

The project also investigates:

- Whether route-level movement features capture information different from traditional athletic measurements.
- Whether the signal remains when players have sufficient NFL opportunities.
- Whether the relationship is consistent across draft classes.
- Whether nonlinear models provide meaningful additional predictive power.
- Whether the observed signal has a plausible movement-based interpretation.

---

## Data

The project uses the following NFL Big Data Bowl datasets:

- `combine_results.csv`
- `combine_tracking.csv`
- `players.csv`
- `player_play.csv`
- `games.csv`
- `game_tracking_2023.csv`
- `game_tracking_2024.csv`
- `game_tracking_2025.csv`

Raw competition data is **not included in this repository** because of file size and competition-data distribution constraints.

The analysis focuses primarily on wide receivers drafted in **2023–2025** for whom Combine movement data can be linked to subsequent NFL performance.

---

## Methodology

### 1. Combine Athletic Features

Conventional Combine measurements are used as the baseline feature set.

The main baseline variables are:

- 40-yard dash time
- Vertical jump

These provide a compact representation of conventional athletic ability.

---

### 2. Whole-Route Movement Features

Rather than examining only isolated moments such as peak speed, the project reconstructs complete Combine routes from tracking data.

The primary drills are:

- `DAGGER_ROUTE_LEFT`
- `COMEBACK_ROUTE_RIGHT`

For each player and route attempt, tracking observations are ordered chronologically and used to calculate movement characteristics such as:

- Route duration
- Path length
- Direct distance
- Straightness ratio
- Peak speed
- Mean speed
- Median speed
- Speed standard deviation
- Speed coefficient of variation
- Early-route speed
- Late-route speed
- Late/early speed ratio

The primary movement feature investigated in the final modeling pipeline is:

**Speed coefficient of variation (CV)**

It measures the relative variability of a player's speed throughout the route:

`Speed CV = standard deviation of route speed / mean route speed`

This captures something that the 40-yard dash does not directly measure: **how consistently a receiver modulates speed throughout an entire route.**

---

## 3. Future NFL Outcome

The primary NFL outcome is **receiving efficiency**, defined as:

`Receiving efficiency = total receiving yards on observed target plays / number of observed target plays`

The outcome is reconstructed from regular-season post-draft NFL play data.

Only target plays are included in the denominator, allowing the metric to focus on production conditional on receiving a target rather than simply measuring opportunity.

Because receiving outcomes are not equally well observed for every player, the analysis also evaluates reliability thresholds based on the number of observed targets.

---

## 4. Player Matching

Combine players are linked to NFL performance using `nfl_id`.

The pipeline:

1. Identifies relevant Combine receiver participants.
2. Extracts whole-route movement features.
3. Matches players to post-draft NFL play data.
4. Filters to regular-season target plays.
5. Calculates future receiving efficiency.
6. Merges Combine, movement, and outcome features.
7. Applies reliability thresholds based on observed target volume.

---

## Modeling

Three primary model configurations are compared:

### Conventional

`40-yard dash + Vertical jump`

### Movement

`Route speed variability`

### Combined

`40-yard dash + Vertical jump + Route speed variability`

The main model is **Ridge regression with standardized features**.

To evaluate whether the conclusions depend on a linear specification, polynomial degree-2 models are also tested.

Validation uses repeated cross-validation with:

- 5 folds
- 10 repeats
- fixed random state for reproducibility

Additional analyses include:

- Leave-one-draft-year-out validation
- Reliability-threshold analysis
- Player-level paired bootstrap comparisons
- Feature correlation analysis
- Residualized movement analysis
- Nonlinear model comparison

---

# Key Results

## 1. Route movement contains a promising incremental signal

The movement-only model generally performs better than the conventional 40-yard dash + vertical baseline when the receiving outcome has sufficient observations.

For players with at least **20 observed targets**:

- Conventional RMSE: approximately **1.88**
- Movement RMSE: approximately **1.78**
- Movement improvement: approximately **5.3%**

The corresponding MAE improvement is approximately **4.7%**.

The direction of the improvement is consistent across several reliability thresholds.

However, the effect is modest and should not be interpreted as proof that movement features universally outperform traditional Combine measurements.

---

## 2. Combining movement with conventional metrics does not automatically improve performance

An important finding is that adding speed variability to the conventional feature set does **not** consistently improve the model.

In several validation settings, the combined model performs slightly worse than the conventional baseline.

This suggests that the useful information in the movement feature should not simply be interpreted as an additional independent predictor that always improves a standard athletic model.

The movement-only signal is more interesting than the naive combined model.

---

## 3. Reliability matters

The predictive signal becomes more favorable as the number of observed NFL targets increases.

This is important because players with very few NFL targets provide noisy estimates of receiving efficiency.

The analysis therefore reports results at multiple thresholds:

- All observed players
- ≥10 observed targets
- ≥20 observed targets
- ≥30 observed targets

The strongest and most stable results occur in players with more reliable NFL outcome measurements.

---

## 4. Nonlinear modeling provides limited additional value

Degree-2 polynomial models were tested to determine whether the movement relationship required nonlinear modeling.

For the ≥20-target population, polynomial movement modeling produced only a small improvement over the corresponding linear movement model, and paired bootstrap confidence intervals crossed zero.

Therefore, the evidence does **not** support adding substantial nonlinear complexity to the final model.

The simpler movement representation is preferable for interpretability.

---

## 5. The movement feature is not simply another 40-yard dash

The correlation between route speed variability and the 40-yard dash is modest.

For the final common population:

- Pearson correlation: approximately **0.20**
- Spearman correlation: approximately **0.11**

This suggests that route-level speed variability is not merely a disguised measurement of straight-line sprint speed.

That distinction is important: the Combine tracking data provides information about **movement behavior across a route**, rather than only maximum straight-line athleticism.

---

# Final Interpretation

The central finding of this project is:

> **Whole-route Combine movement contains a promising but modest incremental signal for future NFL receiving efficiency beyond conventional athletic measurements.**

The signal is strongest when the future NFL outcome is sufficiently observed, while additional nonlinear model complexity provides little extra value.

Importantly, the project does **not** establish that speed variability causes better receiving performance, nor that it should replace conventional Combine metrics.

Instead, it provides evidence that **whole-route movement behavior may contain information that is not fully captured by traditional Combine tests.**

---

# Limitations

Several limitations should be considered.

### Limited sample size

The final modeling population is small because the analysis requires:

- Combine results
- Combine tracking
- player identity matching
- post-draft NFL play data
- observed target outcomes

Only **64 players** enter the locked final modeling population.

### Outcome reliability

Receiving efficiency can be noisy when a player has very few targets. Reliability thresholds are therefore important.

### Missing Combine measurements

Several traditional Combine variables have substantial missingness, particularly:

- Three-cone
- Short shuttle
- Bench press

The final baseline intentionally uses variables with better coverage.

### Tracking coordinate units

The physical units of the Combine tracking coordinates are not independently verified. The analysis therefore emphasizes relative movement characteristics rather than interpreting coordinates as precisely calibrated physical measurements.

### Draft-year coverage

Only players from the available 2023–2025 draft classes are analyzed, limiting the ability to generalize across many NFL generations.

### Association, not causation

The models identify predictive associations. They do not establish that a specific movement pattern causes better NFL receiving performance.

---

# Repository Structure

```text
nfl-big-data-bowl-2027/
│
├── data/
│   └── # Raw competition data (not committed)
│
├── docs/
│   └── # Research notes and documentation
│
├── models/
│   └── # Saved model artifacts
│
├── notebooks/
│   └── # Exploratory and research notebooks
│
├── research_outputs/
│   ├── final_figures/
│   ├── model_comparison/
│   ├── receiving_efficiency_locked/
│   └── nonlinear_final/
│
├── src/
│   ├── analysis/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── outcomes/
│   ├── utils/
│   └── validation/
│
├── submission/
│   └── # Submission-related material
│
├── tests/
│   └── # Validation and testing scripts
│
└── README.md
```

---

# Reproducibility

The analysis was developed in Python using common scientific computing and machine-learning libraries, including:

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib

Raw Kaggle competition files should be placed in the `data/` directory according to the expected filenames.

The analysis scripts under `src/` can then be used to reproduce the feature engineering, outcome reconstruction, validation experiments, and figures.

---

# Main Outputs

The repository contains the outputs used to support the final analysis, including:

- Whole-route movement features
- Reconstructed NFL receiving outcomes
- Locked model comparisons
- Reliability-threshold analyses
- Nonlinear model experiments
- Bootstrap robustness tests
- Final research figures

Key final figures include:

- Model RMSE comparison
- Paired prediction-error differences
- Speed variability versus receiving efficiency

---

# Conclusion

Traditional Combine measurements answer questions such as:

**How fast is a receiver?**

Whole-route tracking allows us to ask a different question:

**How does that receiver move through an entire route?**

This project finds preliminary evidence that route-level speed variability contains useful information about future NFL receiving efficiency that is not fully represented by the 40-yard dash and vertical jump.

The result is not a replacement for conventional scouting metrics. Instead, it suggests a broader direction for Combine analysis:

**Future player evaluation may benefit from treating the Combine not only as a collection of isolated tests, but as a source of detailed movement behavior.**

---

## NFL Big Data Bowl 2027

This repository contains the research code, analysis pipeline, experiments, and supporting outputs for my NFL Big Data Bowl 2027 submission.

**Competition:** NFL Big Data Bowl 2027  
**Focus:** Combine Tracking → NFL Performance  
**Primary Position:** Wide Receivers  
**Primary Movement Feature:** Whole-route speed variability  
**Primary Outcome:** Future NFL receiving efficiency
