# NFL Big Data Bowl 2027 — Research Handoff

> **Purpose:** Preserve the current project understanding and exact research state so a future ChatGPT session can resume without asking me to rebuild or re-explain the dataset.
>
> **Important:** This is a research handoff, not a claim that the current hypothesis has been validated. No predictive or causal result has been established yet.

## 1. Competition objective and constraints

The project is for the **2027 NFL Big Data Bowl**. The challenge is to uncover non-obvious, actionable links between NFL Combine tracking data and future regular-season NFL performance.

Known constraints from the competition instructions shared in the conversation:
- Final write-up: **2,000 words or fewer**.
- Fewer than **10 figures/tables**.
- Explicitly connect Combine tracking to regular-season NFL game performance.
- Include a public Kaggle notebook.
- Deadline noted: **January 6, 2027, at 11:59 PM UTC**.

The final work should offer a defensible insight beyond a conventional Combine leaderboard. Avoid strong claims from small samples, treating opportunity as ability, or presenting correlations as causal effects.

## 2. Research direction

### Initial idea

The initial concept was **“The Hidden Cost of a Cut”** / **Movement Efficiency Under Directional Stress (MEDS)**: use receiver route movement mechanics from Combine tracking to investigate NFL performance, especially separation.

A generic maximum-heading-change/cut detector was tested. It was unstable for a meaningful subset of attempts, sometimes jumping between different phases of a route depending on the smoothing window. Therefore, **do not use the detected maximum-angle cut as a primary feature** or present it as reliable without developing and validating a better event detector.

### Current direction

Use **whole-route movement features** from named receiver route drills, then test whether they add information beyond conventional Combine measurements when explaining or predicting later NFL receiving outcomes.

The primary exploratory route drills are:
- `DAGGER_ROUTE_LEFT`
- `COMEBACK_ROUTE_RIGHT`

This is a working direction, not a final locked hypothesis. Compare a conventional-Combine baseline with a Combine-plus-route-features model. Investigate whether route profiles capture information that straight-line Combine tests miss; do not assume they do.

## 3. Local project location

Project folder:

```text
C:\Users\singh\Downloads\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027\
```

The PowerShell working directory has often been:

```text
C:\Users\singh\Downloads\nfl-big-data-bowl-2027>
```

Python scripts commonly use this fallback:

```python
from pathlib import Path

ROOT = Path("nfl-big-data-bowl-2027")
if not ROOT.exists():
    ROOT = Path(".")
```

Research outputs are saved under `research_outputs\` inside the project directory. Use column selection, chunking, and/or row limits for exploratory inspection; the game-tracking files are large.

## 4. Data files identified

| File | Approximate size / known shape | Relevant contents |
|---|---:|---|
| `combine_results.csv` | ~41 KB; 510 prospects | Combine measurements and NGS scores |
| `combine_tracking.csv` | ~84 MB; 463,189 rows | Combine drill tracking: time, position, speed, acceleration, distance, direction |
| `games.csv` | Small | Game IDs, season, season type, week, teams |
| `game_tracking_2023.csv` | ~335 MB | NFL game tracking |
| `game_tracking_2024.csv` | ~718 MB | NFL game tracking |
| `game_tracking_2025.csv` | ~1.02 GB | NFL game tracking |
| `players.csv` | Small | Player IDs, names, draft year, NFL position, college and draft information |
| `player_career_successes.csv` | Small | Career snaps, active games, starts, All-Pro and Pro Bowl indicators |
| `player_play.csv` | ~147 MB | Play-level player information, targets, receiving/rushing fields, routes and separation |

Relevant fields in `combine_results.csv`:
- `draft_year`, `nfl_id`, `combine_position`
- `combine_height`, `combine_weight`
- `hand_size`, `arm_length`, `wing_span`
- `ten_yd_split`, `forty`, `vertical`, `broad_jump`
- `three_cone`, `short_shuttle`, `bench_reps`
- `ngs_athleticism_score`, `ngs_college_production_score`, `ngs_final_score`

Relevant fields in `combine_tracking.csv`:
- `draft_year`, `event_id`, `nfl_id`, `entity_type`
- `time`, `drill_type`, `drill_name`, `attempt`
- `x`, `y`, `s`, `a`, `dis`, `dir`

Relevant fields in `player_play.csv`:
- `game_id`, `play_id`, `team_abbr`, `nfl_id`, `lined_up_position`
- `target`, `rec_yards`, `rush_yards`, `yards_after_catch`
- `route_ran`, `separation_at_pass_forward`

The play-level analysis so far joins `player_play.csv` to `games.csv` and `players.csv`, restricts to **regular-season (`REG`) plays in seasons on or after the player's draft year**, and focuses on `WR` and `SLOT_WR`.

## 5. Current audit findings

These are descriptive audit results, not performance findings.

### Combine and career coverage
- 510 prospects in Combine results.
- Draft classes: 2023 = 160, 2024 = 177, 2025 = 173.
- Combine position counts: DB = 122, OL = 121, DL = 118, WR = 107, TE = 42.
- All 510 Combine prospects matched to `players.csv` and the career outcome table.
- `combine_tracking.csv`: 463,189 rows, 510 players, approximately 6,310 unique player/drill-attempt events.
- 497 of 510 prospects appeared in at least one of the available 2023–2025 game-tracking files. This is a broad matching result, not proof of equal or complete observation.

### Receiver route tracking
- 107 WR prospects in Combine tracking, with 109,508 rows in the receiver-focused tracking subset.
- `DAGGER_ROUTE_LEFT`: 107 players, 115 attempts, 7,551 rows.
- `COMEBACK_ROUTE_RIGHT`: 98 players, 110 attempts.
- Other drills such as slant and curl also have broad coverage, but the current whole-route feature pipeline focuses on dagger and comeback.
- Typical route attempts have roughly 60 samples at 0.1-second intervals.

### Tracking sanity check
A check of 225 attempts found:
- Sampling is approximately 10 Hz.
- Supplied speed `s` closely agrees with speed derived from x/y changes: median per-attempt correlation was approximately 0.999, with minimum around 0.995.
- Supplied acceleration `a` did **not** consistently agree with the derivative of speed: median correlation approximately -0.153 overall, with route-specific values varying.
- Do not interpret supplied `a` as signed tangential acceleration without further investigation.

### Current route-feature distributions
For 225 route attempts:
- Duration: median ~6.1 seconds; maximum ~14.2.
- Path length: median ~36.573; mean ~34.344; range ~18.639–50.213.
- Direct start-to-end distance: median ~31.195; mean ~27.689.
- Straightness ratio: median ~0.839; mean ~0.791; range ~0.266–0.958.
- Peak speed: median ~8.207.
- Mean speed: median ~5.490.
- Median speed: median ~6.013.
- Speed standard deviation: median ~2.042.
- Speed coefficient of variation: median ~0.368.
- Late/early median-speed ratio: median ~0.961.

**Unit caveat:** The physical scale/units of x/y and derived path length have not been verified. Do not label path length in yards/metres or make physical-unit claims until this is checked. The 14.2-second duration outlier also needs inspection.

## 6. Whole-route features already built

Script: `build_whole_route_features.py`

Outputs:
- `research_outputs/wr_whole_route_features.csv` — 225 attempts.
- `research_outputs/wr_player_route_features.csv` — 205 player-route rows, 107 unique players.

Features calculated per route attempt:
- `duration`
- `n_samples`
- `path_length`
- `direct_distance`
- `straightness_ratio`
- `peak_speed`
- `mean_speed`
- `median_speed`
- `speed_std`
- `speed_cv`
- `early_median_speed`
- `late_median_speed`
- `late_early_speed_ratio`

The player-level route file aggregates route features by player and route. These are whole-route features; the unreliable maximum-heading-change/cut feature is not part of the recommended primary feature set.

After pivoting into the receiver research dataset, route-specific fields include:
- `straightness_ratio_COMEBACK_ROUTE_RIGHT`
- `straightness_ratio_DAGGER_ROUTE_LEFT`
- `peak_speed_COMEBACK_ROUTE_RIGHT`
- `peak_speed_DAGGER_ROUTE_LEFT`
- `mean_speed_COMEBACK_ROUTE_RIGHT`
- `mean_speed_DAGGER_ROUTE_LEFT`
- `speed_cv_COMEBACK_ROUTE_RIGHT`
- `speed_cv_DAGGER_ROUTE_LEFT`
- `late_early_speed_ratio_COMEBACK_ROUTE_RIGHT`
- `late_early_speed_ratio_DAGGER_ROUTE_LEFT`

## 7. Current receiver outcome dataset

Script: `inspect_outcome_merge.py`

Output: `research_outputs/wr_research_dataset.csv`

Known properties:
- Shape: 106 rows × 43 columns.
- 106 unique player IDs; no duplicate IDs.
- WR/SLOT_WR Combine prospects from draft classes 2023–2025.
- Route features are merged by `nfl_id`.
- Career success fields are joined from `player_career_successes.csv`.

Current play-level outcome aggregates include:
- `targets`: count of target rows after filtering to regular season and season >= draft year.
- `mean_separation`: mean of `separation_at_pass_forward` among targeted rows with an observed separation value.
- `observed_receiving_yard_rows`: number of target rows with non-missing `rec_yards`.
- `receiving_yards_observed_sum`: sum of non-missing `rec_yards` values.
- `mean_yards_after_catch`: mean of observed `yards_after_catch`.

**Receiving-yards caveat:** `rec_yards` is missing for many target rows. `receiving_yards_observed_sum` is **not a complete career receiving-yard total** and must not be treated as one.

Known missingness:
- `forty`: ~9.4%
- `vertical`: ~10.4%
- `short_shuttle`: ~64.2%
- `targets`: ~26.4%
- `mean_separation`: ~26.4%
- `receiving_yards_observed_sum`: ~26.4%
- `straightness_ratio_COMEBACK_ROUTE_RIGHT`: ~8.5%
- `straightness_ratio_DAGGER_ROUTE_LEFT`: 0%

WR counts by draft year: 2023 = 38, 2024 = 31, 2025 = 37.

## 8. Outcome coverage audit

Script: `validate_outcome_coverage.py`

Output: `research_outputs/wr_outcome_coverage.csv`

The script joins play-level records to games and player data, applies regular-season/post-draft filters, and reports recorded-play coverage by player. It includes fields such as:
- `recorded_postdraft_plays`
- `observed_target_flags`
- `targets_from_plays`
- `seasons_observed`
- `has_recorded_plays`
- `has_observed_targets`

Known coverage summary:

| Draft year | WR prospects | Players with recorded plays | Players with recorded targets | Median recorded plays per player | Median seasons observed |
|---|---:|---:|---:|---:|---:|
| 2023 | 38 | 31 | 30 | 687 | 3 |
| 2024 | 31 | 28 | 27 | 280 | 2 |
| 2025 | 37 | 23 | 21 | 186 | 1 |

Target counts among players with targets:

| Draft year | Players | Mean targets | Median targets | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| 2023 | 30 | 127.03 | 111.5 | 2 | 432 |
| 2024 | 27 | 71.78 | 39 | 3 | 224 |
| 2025 | 21 | 32.57 | 29 | 3 | 89 |

Exposure and observation windows matter: the 2023 class has had more seasons to accumulate targets than the 2025 class.

**Do not automatically convert missing play rows to zero targets or interpret them as poor performance.** Some players have career active-game or snap counts despite having no rows in the filtered `player_play` subset. Missing rows may reflect coverage/recording differences, not a true zero. Players with only a few targets also have less reliable outcome estimates than players with many targets.

## 9. Critical statistical and research cautions

1. **Outcome denominator matters.** `targets_from_plays` is a target count, but it is not the number of observations supporting `mean_separation`. Separation can be missing on some target rows. Before filtering for reliability, add and inspect a separate count of targeted plays with observed separation (e.g. `separation_count`).
2. **Do not use raw target count as a pure measure of receiver ability.** Targets depend on opportunity, role, team, quarterback, playing time, and other factors.
3. **Do not compare raw career totals across draft years without accounting for exposure.** Classes have different numbers of post-draft seasons observed.
4. **Do not use `receiving_yards_observed_sum` as total receiving yards.** It is incomplete in the present data.
5. **Small sample warning:** there are only about 106 players overall and fewer once outcome-coverage thresholds are applied. Keep models simple and report uncertainty.
6. **Prevent leakage:** every feature and outcome must respect the timing of the prediction question. A Combine-based prediction should use only information available at Combine time as features.
7. **Validate by player and draft year.** Do not randomly split individual plays from the same player between train and test. Consider draft-year-aware validation, while acknowledging that only a few draft classes are available.
8. **Use a baseline.** Compare a conventional Combine-only model with a Combine-plus-route-tracking model. Report whether route features add value beyond the baseline, not merely whether a feature correlates with the outcome.
9. **Sensitivity analysis:** test whether conclusions change with different minimum counts of observed target/separation plays.
10. **Do not claim causality.** The data can support association/prediction unless a credible causal design is established.
11. **Verify route feature definitions and coordinate units.** Inspect duration outliers and verify x/y units before assigning physical units to path length or distance.
12. **No result has yet established that route features predict NFL success.** The project is still in data validation and research design.

## 10. Scripts and workflow status

Scripts already mentioned or run:
- `audit_data.py` — quick file/column audit using limited rows.
- `inspect_research_data.py` — player overlap and route/drill distribution checks.
- `profile_wr_routes.py` — WR route-tracking coverage.
- `build_movement_features.py` — earlier movement-feature attempt; not the preferred final route feature pipeline.
- `check_tracking_units.py` — speed and acceleration sanity checks.
- `build_movement_features_v2.py` — heading-change/cut experiment; unstable and not recommended as a primary feature.
- `build_whole_route_features.py` — current whole-route feature generation.
- `inspect_outcome_merge.py` — creates `wr_research_dataset.csv`.
- `validate_outcome_coverage.py` — creates `wr_outcome_coverage.csv`.

A further script, `prepare_model_dataset.py`, was provided but **has not been confirmed as run**. It is intended to merge the research dataset and outcome coverage, focus on 2023–2024 for a primary comparison, and create:
- `research_outputs/wr_primary_2023_2024.csv`
- `research_outputs/wr_primary_2023_2024_20plus_targets.csv`

It proposes a threshold of at least 20 targets, but this is only a preliminary filter, not a statistically justified final rule. Ideally, supplement or replace it with a threshold on the **number of observed separation values** when modeling `mean_separation`.

## 11. Recommended next session

Continue from existing outputs. **Do not rebuild the dataset or ask the user to rerun all audits** unless an output is missing or a specific inconsistency requires it.

Suggested order:
1. Ask the user to share this README and, if available, the terminal output from `prepare_model_dataset.py`.
2. Confirm the primary files exist and inspect the target-count consistency check.
3. Add an explicit denominator for `mean_separation`: targeted plays with non-missing `separation_at_pass_forward`.
4. Inspect outcome coverage and the long-duration route attempt before modeling.
5. Verify coordinate units and route-feature definitions.
6. Choose the primary outcome and eligibility rule based on data support, not convenience.
7. Start with transparent descriptive plots and uncertainty/reliability checks.
8. Compare a small conventional Combine baseline against a Combine-plus-whole-route feature model, using player-level and draft-year-aware validation where feasible.
9. Run threshold sensitivity analyses and state limitations plainly.
10. Only then settle the final research claim, figures, and notebook narrative.

## 12. How to work with the user

- Give direct, step-by-step instructions for Windows PowerShell and VS Code.
- When code changes are needed, provide a **complete replacement script**, not instructions to locate and edit snippets.
- Avoid dumping large raw datasets or loading giant game-tracking files into memory without a clear reason.
- Do not ask the user to rebuild a dataset that already exists.
- Clearly separate observed results, inference, and unknowns.
- Challenge weak assumptions and unsupported claims. Prioritize a defensible result over a flashy but unstable feature.
- The user may return after a break; use this document as the continuity record.

---

**Current state in one sentence:** Data audits, whole-route feature extraction, receiver outcome merging, and outcome-coverage checks have been completed; the project has **not yet** established a validated research finding or final predictive model.
