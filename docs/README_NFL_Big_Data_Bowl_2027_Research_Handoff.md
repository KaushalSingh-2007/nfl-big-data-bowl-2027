# NFL Big Data Bowl 2027 — Research Handoff

> **Purpose:** Preserve the current project understanding and exact research state so a future ChatGPT session can resume without asking me to rebuild or re-explain the dataset, experiments, decisions, or conclusions.
>
> **Important:** This is a research handoff, not a claim that the current hypothesis has been validated. It deliberately records negative, inconclusive, and abandoned directions so future work does not repeat them.

---

# 1. Competition Objective and Constraints

## Competition objective

The project is based on the NFL Big Data Bowl 2027 theme:

> **Combine to NFL Performance**

The broad research question is:

> Can 10 Hz Combine tracking data reveal non-obvious movement characteristics that are associated with, or improve prediction of, future regular-season NFL performance?

The current work focuses on wide receivers because receiver-specific Combine route drills provide detailed movement trajectories and because receiver-level NFL outcomes can be constructed from regular-season tracking/play data.

The intended research contribution is not simply to reproduce conventional Combine scouting metrics. The goal is to test whether route-level movement information contains additional predictive information beyond traditional Combine measurements.

## Important competition/research constraints

The final project should remain:

- reproducible,
- computationally manageable,
- statistically defensible,
- interpretable,
- temporally correct,
- and explicit about the connection between Combine tracking and future NFL performance.

Exploratory correlations must not be presented as causal or predictive findings without out-of-sample validation.

---

# 2. Research Direction and Hypothesis Evolution

## 2.1 Initial idea — The Hidden Cost of a Cut / MEDS

The original idea was approximately:

> **The Hidden Cost of a Cut** / **Movement Efficiency Under Directional Stress (MEDS)**

The idea was to detect sharp directional changes in Combine tracking and determine whether the ability to execute a cut efficiently predicted later NFL performance.

A maximum-heading-change / candidate-cut detector was developed and investigated.

## 2.2 Why the cut detector was abandoned

The detected turning points were not sufficiently stable.

Candidate angle/time could change depending on:

- local sampling,
- smoothing,
- heading noise,
- threshold choices,
- route geometry,
- and the exact point selected as the cut.

Because of this instability, a single automatically detected "cut" was not considered sufficiently reliable for the primary research claim.

The project therefore moved toward whole-route movement features.

## 2.3 Current direction — whole-route movement

Instead of asking:

> Where exactly did the player cut?

the current approach asks:

> What does the player's complete movement profile look like during a receiver route?

The primary receiver drills are:

- `DAGGER_ROUTE_LEFT`
- `COMEBACK_ROUTE_RIGHT`

The main route-level features include:

- duration,
- number of samples,
- path length,
- direct distance,
- straightness ratio,
- peak speed,
- mean speed,
- median speed,
- speed standard deviation,
- speed coefficient of variation,
- early/late speed,
- late/early speed ratio.

---

# 3. Local Project Location

Current nested project root:

`C:\Users\singh\Downloads\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027`

Outer working directory:

`C:\Users\singh\Downloads\nfl-big-data-bowl-2027`

The nested structure is intentional and is the current working project layout.

---

# 4. Current Repository Structure

```text
nfl-big-data-bowl-2027/
│
├── archive/
│   ├── experiments/
│   └── reports/
│
├── data/
│
├── docs/
│
├── models/
│
├── notebooks/
│
├── research_outputs/
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
│
└── tests/
```

Principle:

- raw data stays in `data/`
- source code stays in `src/`
- generated research outputs stay in `research_outputs/`
- documentation stays in `docs/`
- superseded experiments/reports are archived rather than silently destroyed

---

# 5. Data Files Identified

Current local `data/` files:

```text
combine_results.csv
combine_tracking.csv
game_tracking_2023.csv
game_tracking_2024.csv
game_tracking_2025.csv
games.csv
player_career_successes.csv
player_play.csv
players.csv
```

Approximate local sizes:

- `combine_results.csv` — ~0.04 MB
- `combine_tracking.csv` — ~79.91 MB
- `game_tracking_2023.csv` — ~319.92 MB
- `game_tracking_2024.csv` — ~684.69 MB
- `game_tracking_2025.csv` — ~975.20 MB
- `games.csv` — ~0.04 MB
- `player_career_successes.csv` — ~0.01 MB
- `player_play.csv` — ~140.07 MB
- `players.csv` — ~0.05 MB

The game tracking files are large enough that future scripts should avoid unnecessary full-file scans.

---

# 6. Important Raw Columns

## 6.1 Combine results

```text
draft_year
nfl_id
combine_position
combine_height
combine_weight
hand_size
arm_length
wing_span
ten_yd_split
forty
vertical
broad_jump
three_cone
short_shuttle
bench_reps
```

## 6.2 Combine tracking

```text
draft_year
event_id
nfl_id
entity_type
time
drill_type
drill_name
attempt
x
y
s
a
dis
dir
```

Combine tracking is approximately 10 Hz.

## 6.3 Game tracking

```text
game_id
play_id
nfl_id
time
x
y
s
a
dis
o
dir
event
```

## 6.4 Player play

```text
game_id
play_id
team_abbr
nfl_id
lined_up_position
target
rec_yards
rush_yards
yards_after_catch
route_ran
separation_at_pass_forward
```

## 6.5 Players

```text
nfl_id
display_name
draft_year
nfl_position
birth_date
college_name
college_conference
draft_round
draft_pick_within_round
draft_overall_pick
```

## 6.6 Games

```text
game_id
game_key
season
season_type
week
home_team_abbr
visitor_team_abbr
```

## 6.7 Career outcomes

`player_career_successes.csv` includes:

```text
nfl_id
career_offensive_snaps
career_defensive_snaps
career_special_teams_snaps
career_games_active
career_games_started
ap_all_pro_1st_team
ap_all_pro_2nd_team
pro_bowl_original_ballot
```

These may become useful for a later career-outcome analysis, but exposure/censoring across draft years must be handled carefully.

---

# 7. Current Research Population

Primary research dataset:

`research_outputs/wr_research_dataset.csv`

Current shape:

```text
106 rows × 43 columns
```

Unique players:

```text
106
```

Draft-year distribution:

```text
2023: 38
2024: 31
2025: 37
```

The current population is deliberately narrowed to WRs for the separation analysis.

---

# 8. Current Research Dataset Audit

## Mean separation

Among the 106 research players:

```text
Observed: 78
Missing:  28
```

Observed mean-separation statistics:

```text
Mean:   3.6519
Median: 3.4815
Std:    0.8526
Min:    1.6888
Max:    6.2547
```

Missing separation is **not treated as zero**.

A missing value means that a valid observed player-level separation outcome could not be constructed.

## Targets

Among players with observed separation:

```text
Mean:   82.47
Median: 44
Std:    90.28
Min:    2
Max:    432
```

Target count represents opportunity and should not automatically be interpreted as ability.

## Receiving yards

```text
Mean:   646.96
Median: 319.5
Std:    801.71
Min:    2
Max:    4191
```

## Mean YAC

```text
Mean:   3.8737
Median: 3.7525
Std:    1.7503
Min:    0
Max:    8.7755
```

---

# 9. Combine Feature Missingness

Important missingness in the 106-player research dataset:

```text
bench_reps       73.6%
three_cone       69.8%
short_shuttle    64.2%

draft_round      28.3%
draft_pick       28.3%

mean_separation  26.4%
targets          26.4%

broad_jump       11.3%
vertical         10.4%
forty             9.4%
ten_yd_split      9.4%
```

Several primary movement features have approximately 8.5% missingness.

---

# 10. Complete-Case Constraints

For mean separation:

```text
All 6 major conventional Combine metrics: 19 complete players
All 10 movement features:                70 complete players
All 16 Combine + movement features:      19 complete players
```

This is a major reason the primary model comparison was deliberately simplified.

Three-cone and short-shuttle are too incomplete to include in a fair small-sample primary model without dramatically shrinking the population.

---

# 11. Whole-Route Feature Engineering

Primary route-feature output:

`research_outputs/wr_whole_route_features.csv`

Primary drills:

```text
DAGGER_ROUTE_LEFT
COMEBACK_ROUTE_RIGHT
```

Attempt-level data:

```text
225 attempts
```

Player-route data:

```text
205 player-route rows
107 players
```

Feature set:

```text
duration
n_samples
path_length
direct_distance
straightness_ratio
peak_speed
mean_speed
median_speed
speed_std
speed_cv
early/late median speed
late_early_speed_ratio
```

---

# 12. Feature Definitions

## Duration

Elapsed time over the tracked route segment.

## Path length

Total distance traveled along the tracked trajectory.

## Direct distance

Straight-line distance between the first and final tracked positions.

## Straightness ratio

Approximately:

`direct_distance / path_length`

Closer to 1 means a more direct route.

## Peak speed

Maximum supplied tracking speed.

## Mean speed

Average supplied tracking speed.

## Speed standard deviation

Variation in speed around the route mean.

## Speed coefficient of variation

Approximately:

`speed_std / mean_speed`

This became the most interesting movement feature during exploration.

## Late/early speed ratio

Compares later-route speed to earlier-route speed.

---

# 13. Tracking Validation

## Sampling rate

Combine tracking is approximately 10 Hz, with observations generally separated by about 0.1 seconds.

## Supplied speed versus x/y-derived speed

The supplied tracking speed was compared with speed reconstructed from positional changes.

Agreement was extremely high.

For the investigated speed-CV outlier:

```text
Correlation ≈ 0.997
```

Therefore the unusual speed-CV value was not caused by an obvious disagreement between supplied speed and trajectory-derived speed.

## Acceleration caveat

The supplied `a` field does not consistently behave like a simple derivative of supplied speed.

Therefore the project should not make strong physical acceleration claims from `a` without further validation.

## Coordinate-unit caveat

The physical units of x/y coordinates have not been independently verified.

Avoid unnecessary statements such as exact physical distances unless coordinate units are explicitly established.

---

# 14. Exploratory Correlations with Mean Separation

Initial correlations included:

```text
short_shuttle                         +0.343
ten_yd_split                          -0.242
peak_speed_COMEBACK_ROUTE_RIGHT       +0.195
three_cone                            -0.176
vertical                              +0.160
broad_jump                            +0.132
speed_cv_DAGGER_ROUTE_LEFT            +0.154
speed_cv_COMEBACK_ROUTE_RIGHT         +0.131
straightness_ratio_COMEBACK_ROUTE_RIGHT +0.111
straightness_ratio_DAGGER_ROUTE_LEFT   +0.111
peak_speed_DAGGER_ROUTE_LEFT           +0.095
mean_speed_DAGGER_ROUTE_LEFT           -0.082
forty                                  -0.094
```

These are exploratory associations only.

They are not evidence of causal effects or robust predictive value.

---

# 15. Initial Model Comparison

Initial models:

### Conventional baseline

```text
40 yd + vertical
```

### Movement-only

```text
Dagger peak speed + Dagger speed CV
```

### Combined

```text
40 yd + vertical
+ Dagger peak speed
+ Dagger speed CV
```

Common population:

```text
64 players
```

Initial 5-fold shuffled CV:

```text
Conventional:
RMSE 0.8547
MAE  0.6648
R²   -0.0957

Movement:
RMSE 0.8392
MAE  0.6577
R²   -0.0497

Combined:
RMSE 0.8649
MAE  0.6848
R²   -0.1314
```

The initial movement-only result looked slightly better.

However, this comparison followed feature/configuration exploration.

Therefore it was treated as exploratory rather than final evidence.

---

# 16. Movement Feature Diagnostic

Four Dagger movement features and four Ridge regularization values were explored.

Features:

```text
peak_speed_DAGGER_ROUTE_LEFT
speed_cv_DAGGER_ROUTE_LEFT
straightness_ratio_DAGGER_ROUTE_LEFT
late_early_speed_ratio_DAGGER_ROUTE_LEFT
```

Best exploratory configuration:

```text
speed_cv only
Ridge alpha = 1
```

RMSE:

```text
≈ 0.8240
```

This looked approximately 3.6% better than the earlier conventional baseline.

However, because the feature and alpha were selected after examining the data, the result was recognized as selection-biased.

This directly motivated the locked validation.

---

# 17. Speed CV Investigation

The primary movement feature became:

`speed_cv_DAGGER_ROUTE_LEFT`

Common complete population:

```text
64 players
```

Statistics:

```text
Mean:   0.317965
Std:    0.056219
Min:    0.253226
Median: 0.307343
Max:    0.675295
```

Quantiles:

```text
1%   0.261726
5%   0.274422
10%  0.284463
25%  0.295746
50%  0.307343
75%  0.322343
90%  0.343677
95%  0.376588
99%  0.550362
```

Correlations:

```text
Speed CV vs forty:
Pearson  +0.2012
Spearman +0.1114

Speed CV vs vertical:
Pearson  +0.1141
Spearman +0.0363

Speed CV vs mean separation:
Pearson  +0.1625
Spearman +0.1524
```

These are weak relationships.

---

# 18. Speed CV Outlier Investigation

The maximum speed-CV player was:

```text
nfl_id        56248
draft_year    2023
speed_cv      0.675295
forty         4.59
vertical      39.5
mean_separation 3.921045
```

Raw tracking inspection found:

```text
143 rows
~14.2 seconds
attempt = 1
one event_id
```

The raw `time` column was an ISO timestamp string.

After correctly parsing timestamps and deriving elapsed time:

```text
population speed CV ≈ 0.6783
sample speed CV     ≈ 0.6807
```

Speed and x/y-derived speed:

```text
Correlation ≈ 0.99745
```

The speed trajectory showed legitimate acceleration and deceleration.

Conclusion:

> Player 56248 is a genuine high speed-CV observation and should not be removed merely because it is an outlier.

Raw diagnostic output was saved under:

`research_outputs/model_comparison/speed_cv_signal/outlier_diagnostic/`

---

# 19. Locked Validation Design

A fixed three-model test was created to eliminate feature-search bias.

Common population:

```text
64 players
```

Draft-year composition:

```text
2023: 23
2024: 25
2025: 16
```

## Model A — Conventional

```text
40 yd + vertical
```

## Model B — Movement

```text
speed_cv_DAGGER_ROUTE_LEFT
```

## Model C — Combined

```text
40 yd + vertical + speed_cv_DAGGER_ROUTE_LEFT
```

All models:

```text
StandardScaler
Ridge
alpha = 1.0
```

Validation:

```text
5 folds
10 repeats
50 test folds total
random_state = 42
```

---

# 20. Locked Repeated-CV Results

## A — Conventional

```text
RMSE = 0.8555 ± 0.2230
MAE  = 0.6839 ± 0.1643
R²   = -0.1858 ± 0.1689
```

## B — Movement

```text
RMSE = 0.8952 ± 0.2531
MAE  = 0.7138 ± 0.1708
R²   = -0.4374 ± 1.0757
```

## C — Combined

```text
RMSE = 0.9148 ± 0.2527
MAE  = 0.7261 ± 0.1762
R²   = -0.4812 ± 1.0113
```

The movement-only and combined models are worse on average.

---

# 21. Paired Fold Comparison

Movement versus conventional:

```text
RMSE difference = +0.0397 ± 0.2069
MAE difference  = +0.0299
R² difference   = -0.2517
```

Movement improved:

```text
28 folds
```

and was worse in:

```text
22 folds
```

Combined versus conventional:

```text
RMSE difference = +0.0593 ± 0.1911
MAE difference  = +0.0421
R² difference   = -0.2955
```

---

# 22. Overall Locked Conclusion

Relative to the conventional baseline:

```text
Movement RMSE improvement = -4.63%
Combined RMSE improvement = -6.93%
```

The negative percentages mean worse RMSE.

Therefore:

> The locked repeated-CV analysis does not support the claim that speed CV improves general prediction of future mean separation.

The earlier apparent ~3.6% improvement was selection-biased exploratory evidence and did not survive the locked comparison.

---

# 23. Leave-One-Draft-Year-Out Validation

A separate validation held out each draft year.

## Hold out 2023

Train:

```text
41
```

Test:

```text
23
```

Results:

```text
Conventional:
RMSE 0.8986
MAE  0.6381
R²   -0.3316

Movement:
RMSE 1.0567
MAE  0.7346
R²   -0.8415

Combined:
RMSE 1.0895
MAE  0.7668
R²   -0.9577
```

Movement and combined perform substantially worse.

## Hold out 2024

Train:

```text
39
```

Test:

```text
25
```

Results:

```text
Conventional:
RMSE 0.9145
MAE  0.7491
R²   -0.3409

Movement:
RMSE 0.8704
MAE  0.6900
R²   -0.2148

Combined:
RMSE 0.8921
MAE  0.7294
R²   -0.2760
```

Movement performs somewhat better.

## Hold out 2025

Train:

```text
48
```

Test:

```text
16
```

Results:

```text
Conventional:
RMSE 0.9241
MAE  0.7094
R²   +0.0108

Movement:
RMSE 0.9190
MAE  0.7266
R²   +0.0218

Combined:
RMSE 0.9039
MAE  0.7080
R²   +0.0536
```

The year-specific results vary.

Because each test set is small, these should not be presented as proof of a year-specific effect.

---

# 24. Separation Outcome Reconstruction and Audit

The outcome was independently reconstructed using:

- `player_play.csv`
- `players.csv`
- `games.csv`

Relevant filters:

```text
nfl_position == WR
season_type == REG
post-draft observations
target == 1
separation_at_pass_forward is non-missing
```

The reconstructed observed separation count was compared with the `targets` field in the research dataset.

Result:

```text
78 / 78 players matched
0 mismatches
```

Therefore the target-count field is internally consistent with the reconstructed observed separation outcome.

---

# 25. Outcome Reliability Populations

Within the locked 64-player population:

```text
All observed separation:        64
At least 10 observed rows:      55
At least 20 observed rows:      47
At least 30 observed rows:      40
```

Draft-year counts:

```text
>= 0:
2023 23
2024 25
2025 16

>= 10:
2023 20
2024 21
2025 14

>= 20:
2023 19
2024 17
2025 11

>= 30:
2023 16
2024 16
2025 8
```

Observed separation rows among the 64:

```text
Mean   76.109
Std    74.05
Min    2
Q25    17.5
Median 43
Q75    123
Max    286
```

This confirms substantial variation in outcome reliability.

---

# 26. Reliability-Robustness Model Results

Repeated 10×5 CV was rerun on reliability-filtered populations.

## At least 10 observed separation rows

```text
Conventional RMSE = 0.7126
Movement RMSE     = 0.7033
Combined RMSE     = 0.7254
```

Movement minus conventional:

```text
RMSE = -0.0093
MAE  = -0.0074
R²   = +0.0324
```

## At least 20 observed separation rows

```text
Conventional RMSE = 0.6819
Movement RMSE     = 0.6714
Combined RMSE     = 0.6919
```

Movement minus conventional:

```text
RMSE = -0.0105
MAE  = -0.0159
R²   = +0.0382
```

## At least 30 observed separation rows

```text
Conventional RMSE = 0.6799
Movement RMSE     = 0.6747
Combined RMSE     = 0.7081
```

Movement minus conventional:

```text
RMSE = -0.0051
MAE  = -0.0002
R²   = +0.0023
```

Interpretation:

- movement-only becomes slightly better under stricter outcome reliability,
- the effect is small,
- the effect does not grow consistently with stricter thresholds,
- the combined model remains worse.

---

# 27. Median Separation Sensitivity

The player-level outcome was also rebuilt using median separation rather than mean separation.

Same 64-player population.

Results:

```text
Conventional RMSE = 0.9448
Movement RMSE     = 0.9727
Combined RMSE     = 0.9894
```

Differences:

```text
Movement vs conventional = +0.0280 RMSE
Combined vs conventional = +0.0446 RMSE
```

Therefore the small movement advantage seen under some mean-separation reliability filters does not reproduce under median separation.

This is an important sensitivity result.

---

# 28. Bootstrap Robustness

A paired bootstrap was run using out-of-fold predictions.

Bootstrap replicates:

```text
10,000
```

The purpose was to estimate uncertainty in model differences rather than relying only on the average CV score.

---

# 29. Bootstrap — At Least 20 Separation Rows

Population:

```text
47 players
```

## Movement vs conventional — RMSE

```text
Difference = -0.011780

95% CI:
[-0.024847, +0.001526]

P(improvement) = 0.9572
```

The interval crosses zero.

Therefore the RMSE improvement is not statistically distinguishable from no improvement.

## Movement vs conventional — MAE

```text
Difference = -0.015511

95% CI:
[-0.028714, -0.001941]

P(improvement) = 0.9886
```

This provides evidence of a small MAE improvement in this particular reliability population.

## Combined vs conventional — RMSE

```text
Difference = +0.010021

95% CI:
[+0.004912, +0.016162]
```

Combined is clearly worse.

## Combined vs conventional — MAE

```text
Difference = +0.008364

95% CI:
[+0.003045, +0.014391]
```

Combined is clearly worse.

---

# 30. Bootstrap — At Least 30 Separation Rows

Population:

```text
40 players
```

## Movement vs conventional — RMSE

```text
Difference = -0.008556

95% CI:
[-0.022776, +0.005868]

P(improvement) = 0.8749
```

The interval crosses zero.

## Movement vs conventional — MAE

```text
Difference = -0.000161

95% CI:
[-0.013895, +0.013974]

P(improvement) = 0.5129
```

No evidence of a meaningful MAE improvement.

## Combined vs conventional — RMSE

```text
Difference = +0.024799

95% CI:
[+0.016322, +0.034357]
```

Clearly worse.

## Combined vs conventional — MAE

```text
Difference = +0.025517

95% CI:
[+0.016361, +0.035653]
```

Clearly worse.

---

# 31. Current Reliability Interpretation

The strongest defensible interpretation is:

> **Route-level speed variability exhibits a small possible association with mean separation among players with sufficiently observed target opportunities, but the improvement is not robust across reliability thresholds or error metrics.**

The combined model is more consistently negative:

> Adding speed CV to 40 yd + vertical does not improve out-of-sample mean-separation prediction and often makes it worse.

---

# 32. Route-Specific Signal Analysis

Because the main movement feature came from `DAGGER_ROUTE_LEFT`, a route-specific analysis was performed.

Common movement features:

```text
speed_cv
peak_speed
mean_speed
straightness_ratio
late_early_speed_ratio
```

The purpose was to determine whether the Dagger speed-CV relationship was unique to that route.

---

# 33. Dagger Route Results

For 64 players:

## Speed CV

```text
Pearson  +0.1625
Spearman +0.1524
p ≈ 0.20
```

## Peak speed

```text
Pearson  +0.0680
Spearman +0.0673
```

## Mean speed

```text
Pearson  -0.0916
Spearman -0.0424
```

## Straightness

```text
Pearson  +0.0346
Spearman +0.2093
```

## Late/early speed ratio

```text
Pearson  +0.0192
Spearman -0.0310
```

No strong Dagger-specific relationship was established.

---

# 34. Comeback Route Results

Common population:

```text
57 players
```

## Speed CV

```text
Pearson  +0.1363
Spearman +0.0377
```

## Peak speed

```text
Pearson  +0.2380
Spearman +0.2642
p ≈ 0.047 for Spearman
```

This is descriptively interesting, but many route/features were examined.

It should not be presented as a confirmed discovery without a predefined multiple-testing strategy and independent validation.

Other Comeback features were weakly related to separation.

---

# 35. Dagger vs Comeback Speed CV

Among 57 players with both route measurements:

```text
Dagger mean CV:
0.319745

Comeback mean CV:
0.436559
```

Mean difference:

```text
Dagger - Comeback = -0.116814
```

Median difference:

```text
-0.119713
```

Relationship between the route difference and separation:

```text
Pearson  +0.0224
Spearman +0.1169
```

Therefore the Dagger-minus-Comeback speed-CV difference does not provide a meaningful separation signal.

---

# 36. Cross-Route Speed CV Relationship

Among players with both route values:

```text
Pearson  ≈ -0.0002
Spearman ≈ +0.3902
p ≈ 0.0027
```

The rank relationship is meaningful enough to suggest that speed variability may have some player-level consistency across routes, even though the raw linear relationship is close to zero.

This is suggestive, not definitive.

---

# 37. Partial Correlations

Controlling for:

```text
40 yd
vertical
```

Dagger speed CV:

```text
Pearson  +0.1478
Spearman +0.1494
```

Comeback speed CV:

```text
Pearson  +0.1108
Spearman +0.0438
```

These remain weak.

The movement feature therefore does not appear to be a strong independent predictor of separation after basic Combine athleticism is controlled.

---

# 38. Route-Normalized Analysis

Route-normalized features were also examined.

The main question was whether relative within-route movement could reveal a hidden signal.

The normalized Dagger-minus-Comeback speed-CV difference showed:

```text
Pearson  ≈ +0.0052
Spearman ≈ +0.1158
```

No meaningful relationship was established.

## Important technical clarification

Z-scoring is an affine transformation.

Therefore Pearson and Spearman correlations within the same population are expected to remain unchanged after simple standardization.

The fact that route-normalized correlation values matched the raw correlations is therefore expected.

The useful result from this analysis is that the cross-route normalized difference did not reveal a hidden separation signal.

---

# 39. Main Research Conclusion

The current evidence does **not** support the strong claim:

> "Combine whole-route movement features improve prediction of future NFL separation."

The evidence instead shows:

1. Some movement features have weak descriptive relationships with separation.
2. The initially promising movement result disappeared under locked repeated validation.
3. Speed CV alone can show a small improvement under some reliability filters.
4. That improvement is not robust across stricter thresholds and metrics.
5. Median separation does not reproduce the movement benefit.
6. Route-specific analysis does not reveal a clear Dagger-specific mechanism.
7. Route-normalized analysis does not reveal a hidden signal.
8. Adding speed CV to conventional 40 yd + vertical generally hurts out-of-sample prediction.
9. The speed-CV outlier was genuine and should not be removed merely for model convenience.

The current defensible conclusion is:

> **Route-level speed variability exhibited a small association with mean separation among players with sufficiently observed target opportunities, but the improvement was not robust across reliability thresholds or error metrics.**

---

# 40. What We Learned

## 40.1 Feature search creates false optimism

The exploratory movement search produced a roughly 3.6% RMSE improvement.

After the feature set was locked before evaluation, that advantage disappeared.

Lesson:

> Feature selection and final validation must be separated.

## 40.2 More features do not automatically help

The combined model performed worse than the conventional model.

Lesson:

> A movement feature can contain signal without providing useful incremental predictive value.

## 40.3 Outcome reliability matters

Player-level mean separation is estimated from very different numbers of observed target opportunities.

Lesson:

> Reliability thresholds must be considered before interpreting player-level separation.

## 40.4 Mean and median can tell different stories

The small mean-separation movement signal disappears under median separation.

Lesson:

> Results sensitive to the aggregation method should not be overinterpreted.

## 40.5 Route type matters

Dagger and Comeback routes have substantially different movement distributions.

Lesson:

> A route feature cannot automatically be treated as a universal movement ability.

## 40.6 Outliers must be investigated before removal

The extreme speed-CV observation was validated against raw tracking.

Lesson:

> Do not remove unusual players just because they hurt a model.

## 40.7 Negative results are useful

The project has ruled out a simpler story:

> "Adding route-level speed variability to standard Combine metrics clearly improves prediction of future separation."

That is a useful research result.

---

# 41. What Should NOT Be Done Next

Do not immediately start another large movement-feature sweep.

Avoid:

- trying dozens of new movement features without a specific hypothesis,
- repeatedly changing the target until a positive result appears,
- removing player 56248 because it is inconvenient,
- adding every Combine variable despite extreme missingness,
- using complex ML models on the current tiny complete-case population,
- reporting exploratory correlations as discoveries,
- treating missing separation as zero,
- allowing future-season information into predictors,
- repeatedly optimizing on the same small sample.

The current movement-to-mean-separation question has been investigated enough to justify moving to the next stage.

---

# 42. Recommended Next Research Stage — Outcome Audit

The next step should be an audit of alternative NFL performance outcomes.

The current separation target is noisy and opportunity-dependent.

Potential outcome families:

## Opportunity

```text
targets
routes run
offensive snaps
route participation
```

These describe how often a player is involved.

## Production

```text
receiving yards
receptions
receiving touchdowns
yards after catch
```

These describe actual output.

## Efficiency

Potential candidates:

```text
yards per route
yards per target
YAC per reception
target rate
```

These attempt to normalize production by opportunity.

## Career outcomes

The career-success file contains:

```text
career_offensive_snaps
career_defensive_snaps
career_special_teams_snaps
career_games_active
career_games_started
ap_all_pro_1st_team
ap_all_pro_2nd_team
pro_bowl_original_ballot
```

These may provide longer-horizon outcomes.

---

# 43. Outcome Audit Criteria

Every candidate outcome should be evaluated on:

## Coverage

How many of the 106 players have a valid outcome?

## Reliability

How much NFL observation is needed for a stable player-level estimate?

## Temporal correctness

Does the outcome occur after the Combine?

## Opportunity dependence

Does the outcome mostly measure:

- ability,
- opportunity,
- team role,
- or a mixture?

## Sample size

Prefer outcomes that preserve a useful number of complete players.

## Interpretability

The final Kaggle reader should be able to understand the outcome easily.

## Leakage

No future information may enter the predictors.

---

# 44. Important Temporal Issue

The current draft classes are:

```text
2023
2024
2025
```

These groups have different amounts of NFL observation time.

2023 players have more possible NFL seasons than 2025 players.

Therefore cumulative career totals are not automatically comparable.

Any career-level target must account for:

- exposure time,
- censoring,
- seasons available,
- and possibly games/snaps available.

A raw cumulative career total should not automatically be treated as a fair target across all draft classes.

---

# 45. Current Scripts

## Analysis

```text
src/analysis/inspect_research_data.py
src/analysis/plot_detected_turns.py
src/analysis/profile_wr_routes.py
```

## Feature engineering

```text
src/features/build_movement_features.py
src/features/build_movement_features_v2.py
src/features/build_whole_route_features.py
```

## Outcome auditing

```text
src/outcomes/audit_separation_consistent.py
src/outcomes/audit_separation_reliability.py
src/outcomes/audit_target_count_discrepancy.py
src/outcomes/inspect_outcome_merge.py
```

## Validation

```text
src/validation/audit_data.py
src/validation/check_tracking_units.py
src/validation/validate_movement_features.py
src/validation/validate_outcome_coverage.py
```

## Modeling

```text
src/models/run_separation_model_comparison.py
src/models/movement_diagnostic.py
src/models/inspect_speed_cv_signal.py
src/models/investigate_speed_cv_outlier.py
src/models/inspect_outlier_raw_data.py
src/models/run_locked_separation_validation.py
src/models/run_separation_reliability_robustness.py
src/models/bootstrap_separation_robustness.py
src/models/analyze_route_specific_signal.py
src/models/analyze_route_normalized_signal.py
```

---

# 46. Important Research Output Locations

Primary research dataset:

`research_outputs/wr_research_dataset.csv`

Whole-route features:

`research_outputs/wr_whole_route_features.csv`

Model comparison:

`research_outputs/model_comparison/`

Locked validation:

`research_outputs/model_comparison/locked_validation/`

Speed-CV signal:

`research_outputs/model_comparison/speed_cv_signal/`

Outlier diagnostic:

`research_outputs/model_comparison/speed_cv_signal/outlier_diagnostic/`

Reliability robustness:

`research_outputs/model_comparison/separation_reliability_robustness/`

Bootstrap:

`research_outputs/model_comparison/separation_reliability_robustness/bootstrap/`

Route-specific:

`research_outputs/model_comparison/route_specific_signal/`

Route-normalized:

`research_outputs/model_comparison/route_normalized_signal/`

---

# 47. Backup and Archive Convention

Before materially changing an existing research script:

1. preserve the previous version,
2. use the archive directory when appropriate,
3. run the updated script,
4. inspect outputs,
5. do not silently replace conclusions.

Archive locations:

```text
archive/experiments/
archive/reports/
```

The project should maintain a reproducible trail of important experiments.

---

# 48. Statistical Guardrails

## Missing outcome is not zero

Missing separation means unobserved outcome.

## Do not optimize on the final result

Feature selection must occur before final evaluation.

## Prefer simple models

The current sample is small.

Ridge is more defensible than complex models for the current hypothesis.

## Use repeated validation

A single train/test split is too unstable.

## Use temporal validation where appropriate

Draft-year holdout is useful because future NFL performance is inherently temporal.

## Report uncertainty

Use:

- RMSE,
- MAE,
- R²,
- fold variability,
- paired differences,
- bootstrap confidence intervals where appropriate.

## Association is not prediction

A correlation does not establish out-of-sample predictive value.

## Prediction is not causation

Nothing currently establishes that a movement characteristic causes NFL success.

---

# 49. Current Decision

Current movement/separation experiment status:

```text
EXPLORATION: COMPLETE
VALIDATION: COMPLETE
ROBUST POSITIVE RESULT: NOT ESTABLISHED
```

Current hypothesis:

> Route-level Combine movement features improve prediction of future NFL separation.

Current evidence:

> **Not supported as a robust predictive improvement.**

The movement analysis has nevertheless produced useful methodological findings about:

- route-level feature stability,
- outcome reliability,
- validation bias,
- route dependence,
- and opportunity-sensitive NFL outcomes.

---

# 50. Recommended Next Session

The next session should begin with:

> **NFL OUTCOME AUDIT**

Not another movement-feature sweep.

Recommended sequence:

### Step 1

Audit all available future NFL outcomes for the 106-player WR population.

### Step 2

Calculate coverage and missingness.

### Step 3

Measure opportunity dependence.

### Step 4

Check temporal comparability across 2023/2024/2025.

### Step 5

Identify 2–4 promising outcome candidates.

### Step 6

Choose a primary outcome using a predefined rule.

### Step 7

Only then test whether Combine movement features add predictive value.

This avoids repeated feature hunting against one noisy outcome.

---

# 51. Possible Final Research Story

If no strong positive movement result appears under better outcomes, the project can still have a defensible analytical narrative:

> Combine tracking contains detailed movement information that is not represented by traditional Combine measurements. We extracted whole-route movement characteristics from receiver-specific drills and tested whether these features improve prediction of future NFL performance. Several movement features showed weak descriptive relationships with NFL outcomes, but these relationships did not translate into robust incremental predictive performance under locked repeated validation, reliability filtering, and temporal checks. The analysis therefore demonstrates both the potential and limitations of using Combine tracking for future-performance prediction.

The final submission can emphasize:

- methodology,
- feature engineering,
- validation,
- route-level movement analysis,
- outcome quality,
- and what the negative result teaches.

---

# 52. Current File/Script State

Important scripts currently present include:

```text
src/analysis/inspect_research_data.py
src/analysis/plot_detected_turns.py
src/analysis/profile_wr_routes.py

src/features/build_movement_features.py
src/features/build_movement_features_v2.py
src/features/build_whole_route_features.py

src/outcomes/audit_separation_consistent.py
src/outcomes/audit_separation_reliability.py
src/outcomes/audit_target_count_discrepancy.py
src/outcomes/inspect_outcome_merge.py

src/validation/audit_data.py
src/validation/check_tracking_units.py
src/validation/validate_movement_features.py
src/validation/validate_outcome_coverage.py

src/models/run_separation_model_comparison.py
src/models/movement_diagnostic.py
src/models/inspect_speed_cv_signal.py
src/models/investigate_speed_cv_outlier.py
src/models/inspect_outlier_raw_data.py
src/models/run_locked_separation_validation.py
src/models/run_separation_reliability_robustness.py
src/models/bootstrap_separation_robustness.py
src/models/analyze_route_specific_signal.py
src/models/analyze_route_normalized_signal.py
```

The major outputs described in this document should be considered the current research record.

---

# 53. How to Continue in a New ChatGPT Session

When this handoff is supplied to a future ChatGPT session:

1. Do not restart the project from scratch.
2. Do not repeat the abandoned cut-detector investigation unless specifically requested.
3. Do not repeat the exploratory speed-CV feature search.
4. Do not remove the 56248 outlier without a new, explicit reason.
5. Do not claim that movement features improve separation.
6. Treat the locked validation and robustness results as the current evidence.
7. Start from the outcome audit.
8. Preserve the current repository structure.
9. Prefer exact Windows PowerShell commands when creating/running files.
10. When generating scripts, make them reproducible and computationally aware of the large tracking files.

The next analytical task is:

> **Audit alternative NFL outcomes and determine whether a more reliable, less opportunity-sensitive target can support a stronger Combine-to-NFL research question.**

---

# 54. Current State in One Sentence

> **The project has built and validated a WR Combine-to-NFL research pipeline, extracted whole-route movement features, rigorously tested their relationship with future NFL mean separation, found no robust incremental predictive benefit from the current speed-CV feature, and is now ready to move from movement-feature hunting to a broader audit of stronger NFL outcome targets.**

---

# 55. Final Status Checklist

```text
[COMPLETE] Raw data discovery and audit

[COMPLETE] WR research population construction

[COMPLETE] Whole-route movement feature extraction

[COMPLETE] Tracking consistency checks

[COMPLETE] Mean separation outcome construction

[COMPLETE] Outcome coverage audit

[COMPLETE] Initial movement model exploration

[COMPLETE] Speed-CV signal investigation

[COMPLETE] Speed-CV outlier investigation

[COMPLETE] Locked repeated-CV validation

[COMPLETE] Leave-one-draft-year-out validation

[COMPLETE] Separation reliability robustness

[COMPLETE] Bootstrap robustness

[COMPLETE] Median-separation sensitivity

[COMPLETE] Route-specific analysis

[COMPLETE] Route-normalized analysis

[NEXT] NFL outcome audit

[NEXT] Select stronger primary outcome

[NEXT] Re-evaluate Combine-to-NFL predictive value

[NEXT] Build final competition narrative
```

---

**End of Research Handoff**
