from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "research_outputs"
OUT.mkdir(parents=True, exist_ok=True)

RESEARCH = ROOT / "research_outputs" / "wr_research_dataset.csv"
PLAY = DATA / "player_play.csv"
PLAYERS = DATA / "players.csv"
GAMES = DATA / "games.csv"
CAREER = DATA / "player_career_successes.csv"

print("=" * 80)
print("NFL OUTCOME AUDIT")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Research population
# ---------------------------------------------------------------------
research = pd.read_csv(RESEARCH)

if "nfl_id" not in research.columns:
    raise ValueError("wr_research_dataset.csv does not contain nfl_id")

research["nfl_id"] = pd.to_numeric(research["nfl_id"], errors="coerce")
research = research.dropna(subset=["nfl_id"]).copy()
research["nfl_id"] = research["nfl_id"].astype(int)

population = research[["nfl_id"]].drop_duplicates().copy()

# Prefer the research dataset's draft year if available.
if "draft_year" in research.columns:
    pop_year = research[["nfl_id", "draft_year"]].drop_duplicates("nfl_id")
else:
    pop_year = pd.DataFrame(columns=["nfl_id", "draft_year"])

# ---------------------------------------------------------------------
# 2. Player metadata
# ---------------------------------------------------------------------
players = pd.read_csv(
    PLAYERS,
    usecols=lambda c: c in [
        "nfl_id",
        "display_name",
        "draft_year",
        "nfl_position"
    ]
)

players["nfl_id"] = pd.to_numeric(players["nfl_id"], errors="coerce")
players = players.dropna(subset=["nfl_id"]).copy()
players["nfl_id"] = players["nfl_id"].astype(int)

players = players.drop_duplicates("nfl_id")

population = population.merge(
    players,
    on="nfl_id",
    how="left",
    suffixes=("", "_player")
)

if "draft_year_player" in population.columns:
    population["draft_year"] = population["draft_year"].fillna(
        population["draft_year_player"]
    )
    population.drop(columns=["draft_year_player"], inplace=True)

# ---------------------------------------------------------------------
# 3. Games metadata
# ---------------------------------------------------------------------
games = pd.read_csv(
    GAMES,
    usecols=lambda c: c in ["game_id", "season", "season_type"]
)

games["game_id"] = games["game_id"].astype(str)
games["season"] = pd.to_numeric(games["season"], errors="coerce")

# ---------------------------------------------------------------------
# 4. Player-play data
# ---------------------------------------------------------------------
play_cols = [
    "game_id",
    "play_id",
    "nfl_id",
    "target",
    "rec_yards",
    "yards_after_catch",
    "route_ran",
    "separation_at_pass_forward"
]

plays = pd.read_csv(
    PLAY,
    usecols=lambda c: c in play_cols,
    low_memory=False
)

plays["nfl_id"] = pd.to_numeric(plays["nfl_id"], errors="coerce")
plays = plays.dropna(subset=["nfl_id"]).copy()
plays["nfl_id"] = plays["nfl_id"].astype(int)

plays["game_id"] = plays["game_id"].astype(str)

for col in [
    "target",
    "rec_yards",
    "yards_after_catch",
    "separation_at_pass_forward"
]:
    if col in plays.columns:
        plays[col] = pd.to_numeric(plays[col], errors="coerce")

# Restrict immediately to the 106-player research population.
plays = plays[plays["nfl_id"].isin(population["nfl_id"])].copy()

print(f"Research population: {len(population)} players")
print(f"Player-play rows after player restriction: {len(plays):,}")

# ---------------------------------------------------------------------
# 5. Add season and apply the research outcome definition:
#    regular-season NFL performance in/after the player's draft year.
# ---------------------------------------------------------------------
plays = plays.merge(
    games,
    on="game_id",
    how="left"
)

plays = plays.merge(
    population[["nfl_id", "draft_year"]],
    on="nfl_id",
    how="left"
)

plays["season_type"] = plays["season_type"].astype(str).str.upper()

valid = (
    (plays["season_type"] == "REG") &
    plays["season"].notna() &
    plays["draft_year"].notna() &
    (plays["season"] >= plays["draft_year"])
)

plays = plays.loc[valid].copy()

print(f"Post-draft regular-season player-play rows: {len(plays):,}")

# ---------------------------------------------------------------------
# 6. Basic outcome flags
# ---------------------------------------------------------------------
plays["is_target"] = plays["target"].eq(1)

plays["has_route"] = (
    plays["route_ran"].notna() &
    plays["route_ran"].astype(str).str.strip().ne("")
)

plays["has_separation"] = (
    plays["is_target"] &
    plays["separation_at_pass_forward"].notna()
)

plays["has_rec_yards"] = (
    plays["is_target"] &
    plays["rec_yards"].notna()
)

plays["has_yac"] = (
    plays["is_target"] &
    plays["yards_after_catch"].notna()
)

# ---------------------------------------------------------------------
# 7. Per-player audit table
# ---------------------------------------------------------------------
rows = []

for nfl_id, g in plays.groupby("nfl_id", sort=False):
    row = {
        "nfl_id": int(nfl_id),

        "recorded_postdraft_reg_plays": len(g),
        "seasons_observed": g["season"].nunique(),
        "games_observed": g["game_id"].nunique(),

        "routes_observed": int(g["has_route"].sum()),
        "targets": int(g["is_target"].sum()),

        "observed_separation_count": int(g["has_separation"].sum()),
        "mean_separation": (
            g.loc[g["has_separation"], "separation_at_pass_forward"].mean()
            if g["has_separation"].any() else np.nan
        ),
        "median_separation": (
            g.loc[g["has_separation"], "separation_at_pass_forward"].median()
            if g["has_separation"].any() else np.nan
        ),

        "observed_rec_yard_rows": int(g["has_rec_yards"].sum()),
        "receiving_yards_observed_sum": (
            g.loc[g["has_rec_yards"], "rec_yards"].sum()
            if g["has_rec_yards"].any() else np.nan
        ),

        "observed_yac_rows": int(g["has_yac"].sum()),
        "mean_yac": (
            g.loc[g["has_yac"], "yards_after_catch"].mean()
            if g["has_yac"].any() else np.nan
        ),
    }

    rows.append(row)

audit = pd.DataFrame(rows)

# Make sure players with zero recorded outcomes are retained.
audit = population[["nfl_id"]].merge(
    audit,
    on="nfl_id",
    how="left"
)

# ---------------------------------------------------------------------
# 8. Exposure-normalized quantities
# ---------------------------------------------------------------------
audit["targets_per_route"] = np.where(
    audit["routes_observed"] > 0,
    audit["targets"] / audit["routes_observed"],
    np.nan
)

audit["targets_per_observed_season"] = np.where(
    audit["seasons_observed"] > 0,
    audit["targets"] / audit["seasons_observed"],
    np.nan
)

audit["routes_per_observed_season"] = np.where(
    audit["seasons_observed"] > 0,
    audit["routes_observed"] / audit["seasons_observed"],
    np.nan
)

audit["plays_per_observed_season"] = np.where(
    audit["seasons_observed"] > 0,
    audit["recorded_postdraft_reg_plays"] /
    audit["seasons_observed"],
    np.nan
)

audit["observed_yards_per_observed_target"] = np.where(
    audit["observed_rec_yard_rows"] > 0,
    audit["receiving_yards_observed_sum"] /
    audit["observed_rec_yard_rows"],
    np.nan
)

# Reliability flags.
for threshold in [0, 5, 10, 20, 30, 50]:
    audit[f"separation_ge_{threshold}"] = (
        audit["observed_separation_count"] >= threshold
    )

# ---------------------------------------------------------------------
# 9. Career outcomes
# ---------------------------------------------------------------------
if CAREER.exists():
    career = pd.read_csv(CAREER)

    if "nfl_id" in career.columns:
        career["nfl_id"] = pd.to_numeric(
            career["nfl_id"], errors="coerce"
        )
        career = career.dropna(subset=["nfl_id"]).copy()
        career["nfl_id"] = career["nfl_id"].astype(int)
        career = career.drop_duplicates("nfl_id")

        career_cols = [
            "nfl_id",
            "career_offensive_snaps",
            "career_defensive_snaps",
            "career_special_teams_snaps",
            "career_games_active",
            "career_games_started",
            "ap_all_pro_1st_team",
            "ap_all_pro_2nd_team",
            "pro_bowl_original_ballot"
        ]

        career_cols = [
            c for c in career_cols if c in career.columns
        ]

        audit = audit.merge(
            career[career_cols],
            on="nfl_id",
            how="left"
        )

# Add player metadata.
meta_cols = [
    c for c in ["nfl_id", "display_name", "draft_year", "nfl_position"]
    if c in population.columns
]

audit = audit.merge(
    population[meta_cols].drop_duplicates("nfl_id"),
    on="nfl_id",
    how="left",
    suffixes=("", "_meta")
)

# ---------------------------------------------------------------------
# 10. Candidate outcome summary
# ---------------------------------------------------------------------
candidate_specs = {
    "recorded_postdraft_reg_plays": "Recorded regular-season player-play rows",
    "seasons_observed": "Seasons observed",
    "games_observed": "Games observed",
    "routes_observed": "Observed routes",
    "targets": "Targets",
    "observed_separation_count": "Observed separation rows",
    "mean_separation": "Mean separation",
    "median_separation": "Median separation",
    "observed_rec_yard_rows": "Observed receiving-yard rows",
    "receiving_yards_observed_sum": "Observed receiving yards",
    "observed_yac_rows": "Observed YAC rows",
    "mean_yac": "Mean YAC",
    "targets_per_route": "Targets per observed route",
    "targets_per_observed_season": "Targets per observed season",
    "routes_per_observed_season": "Routes per observed season",
    "plays_per_observed_season": "Recorded plays per observed season",
    "observed_yards_per_observed_target": "Observed yards per observed target",
    "career_offensive_snaps": "Career offensive snaps",
    "career_games_active": "Career games active",
    "career_games_started": "Career games started",
    "ap_all_pro_1st_team": "AP All-Pro 1st team",
    "ap_all_pro_2nd_team": "AP All-Pro 2nd team",
    "pro_bowl_original_ballot": "Pro Bowl original ballot"
}

summary_rows = []

for col, label in candidate_specs.items():
    if col not in audit.columns:
        continue

    s = pd.to_numeric(audit[col], errors="coerce")
    valid_s = s.dropna()

    summary_rows.append({
        "outcome": col,
        "description": label,
        "n_valid": int(valid_s.shape[0]),
        "n_missing": int(s.isna().sum()),
        "missing_pct": float(s.isna().mean() * 100),
        "mean": valid_s.mean() if len(valid_s) else np.nan,
        "median": valid_s.median() if len(valid_s) else np.nan,
        "std": valid_s.std() if len(valid_s) > 1 else np.nan,
        "min": valid_s.min() if len(valid_s) else np.nan,
        "max": valid_s.max() if len(valid_s) else np.nan,
    })

candidate_summary = pd.DataFrame(summary_rows)

# ---------------------------------------------------------------------
# 11. Coverage by draft year
# ---------------------------------------------------------------------
year_rows = []

for year, g in audit.groupby("draft_year", dropna=False):
    for col, label in candidate_specs.items():
        if col not in g.columns:
            continue

        s = pd.to_numeric(g[col], errors="coerce")

        year_rows.append({
            "draft_year": year,
            "outcome": col,
            "description": label,
            "population_n": len(g),
            "valid_n": int(s.notna().sum()),
            "missing_n": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100),
            "mean": s.mean(),
            "median": s.median(),
        })

coverage_by_year = pd.DataFrame(year_rows)

# ---------------------------------------------------------------------
# 12. Separation reliability table
# ---------------------------------------------------------------------
reliability_rows = []

for threshold in [0, 5, 10, 20, 30, 50, 100]:
    mask = audit["observed_separation_count"] >= threshold

    subset = audit.loc[mask]

    reliability_rows.append({
        "minimum_observed_separation_rows": threshold,
        "n_players": int(mask.sum()),
        "draft_2023_n": int(
            ((audit["draft_year"] == 2023) & mask).sum()
        ),
        "draft_2024_n": int(
            ((audit["draft_year"] == 2024) & mask).sum()
        ),
        "draft_2025_n": int(
            ((audit["draft_year"] == 2025) & mask).sum()
        ),
        "mean_separation_mean": subset["mean_separation"].mean(),
        "mean_separation_median": subset["mean_separation"].median(),
        "median_separation_mean": subset["median_separation"].mean(),
        "median_separation_median": subset["median_separation"].median(),
    })

reliability = pd.DataFrame(reliability_rows)

# ---------------------------------------------------------------------
# 13. Data-quality checks
# ---------------------------------------------------------------------
quality = pd.DataFrame([
    {
        "check": "Research population size",
        "value": len(population)
    },
    {
        "check": "Players with post-draft regular-season rows",
        "value": int(audit["recorded_postdraft_reg_plays"].notna().sum())
    },
    {
        "check": "Players with at least one target",
        "value": int((audit["targets"].fillna(0) > 0).sum())
    },
    {
        "check": "Players with observed separation",
        "value": int(audit["observed_separation_count"].fillna(0).gt(0).sum())
    },
    {
        "check": "Players with observed receiving yards",
        "value": int(audit["observed_rec_yard_rows"].fillna(0).gt(0).sum())
    },
    {
        "check": "Players with observed YAC",
        "value": int(audit["observed_yac_rows"].fillna(0).gt(0).sum())
    },
])

# ---------------------------------------------------------------------
# 14. Save outputs
# ---------------------------------------------------------------------
audit_path = OUT / "nfl_outcome_player_audit.csv"
summary_path = OUT / "nfl_outcome_candidate_summary.csv"
coverage_path = OUT / "nfl_outcome_coverage_by_draft_year.csv"
reliability_path = OUT / "nfl_outcome_separation_reliability.csv"
quality_path = OUT / "nfl_outcome_quality_checks.csv"

audit.to_csv(audit_path, index=False)
candidate_summary.to_csv(summary_path, index=False)
coverage_by_year.to_csv(coverage_path, index=False)
reliability.to_csv(reliability_path, index=False)
quality.to_csv(quality_path, index=False)

# ---------------------------------------------------------------------
# 15. Console report
# ---------------------------------------------------------------------
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 180)

print("\n" + "=" * 80)
print("CANDIDATE OUTCOME SUMMARY")
print("=" * 80)
print(candidate_summary.to_string(index=False))

print("\n" + "=" * 80)
print("SEPARATION RELIABILITY")
print("=" * 80)
print(reliability.to_string(index=False))

print("\n" + "=" * 80)
print("COVERAGE BY DRAFT YEAR")
print("=" * 80)

important = [
    "targets",
    "observed_separation_count",
    "mean_separation",
    "median_separation",
    "routes_observed",
    "receiving_yards_observed_sum",
    "mean_yac",
    "career_offensive_snaps",
    "career_games_active",
    "career_games_started"
]

print(
    coverage_by_year[
        coverage_by_year["outcome"].isin(important)
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("QUALITY CHECKS")
print("=" * 80)
print(quality.to_string(index=False))

print("\nSaved:")
for p in [
    audit_path,
    summary_path,
    coverage_path,
    reliability_path,
    quality_path
]:
    print(" -", p)

print("\nDONE")

