# Issue #16: feature options and edge-case checks

## Objective and scope

25 October 2026: Week 1 of Issue #16. Prepare reusable feature options (age, missing-value indicators, location) for the Sprint 2 comparisons, with tests for the edge cases named in the issue. No model was trained. No comparison results exist yet.

## Changed and created

- `src/team_wtm/features.py`: feature groups, named feature options, `add_feature_columns()` and `select_features()`.
- `tests/test_features.py`: 8 tests on a 4-row synthetic table.
- This task log.

No shared cleaning code was changed.

## Decisions and reasoning

- **Reuse shared rules.** Age and quality flags come from `export_cleaned.add_quality_features()`. There is no second age or coordinate implementation (issue rule 1, D03, D04, D11).
- **Coordinate screen.** Uses the audit rectangle from the shared flags (D11: 1,812 failures), not the geospatial notebook's range (1,819). Rows with unusable coordinates are kept. Only the new `longitude_screened` / `latitude_screened` columns are set to missing; the originals are unchanged.
- **Feature groups.** Related columns are switched on or off together. `quantity` and `quantity_group` are in one group because they are identical (D09). This lets the quantity test (D26) remove both.
- **Feature ladder.** `base` → `base_age` → `base_age_missing` → `full` adds one group at a time, so each step shows the effect of one group. `full_no_quantity` is for the quantity test. `single_quantity` and `single_waterpoint_type` are single-field setups for H6, chosen from the strongest Cramér's V results in Issue #5.
- **New flag `gps_height_zero`.** About a third of heights are exactly 0 and are likely unrecorded (D07, D18). Values are kept; the flag lets the model compare with and without this information.
- **Columns.** The shorter `_group` / `_class` versions are used to avoid entering the same information several times. `permit` is kept (D16).
- The `base` list is a proposal and needs agreement with PM-4.

## Checks and results

**Tests:** 8 passed. A deliberate change (construction year 2013 → 2012) made `test_zero_age_is_valid` fail as expected, then it was reverted.

**Real training data (59,400 rows):** every option returned 59,400 rows.

| Flag | Rows | Matches Sprint 1 |
|---|---|---|
| pump_age_missing | 20,718 | 20,709 unknown years + 9 negative ages (D02, D04) |
| pump_age_invalid | 9 | D04 |
| population_missing | 21,381 | D02 |
| gps_height_zero | 20,438 | D18 (59,400 − 38,962) |
| coordinate_unusable | 1,812 | D11 audit rule |

588 pumps have a valid age of 0.

**Missing age is concentrated by region.** Dodoma, Kagera, Mbeya and Tabora have 100% missing age; Shinyanga 96.7%; Mwanza 88.6%. All other regions are 0.2–7.3%. These 6 regions hold about 95% of missing ages.

## Uncertainty and remaining work

- Because missing age is mostly regional, `pump_age_missing` largely encodes region. The known/unknown-age comparison will mostly compare these 6 regions with the rest, and age (H1) cannot be assessed in 4 regions. To be discussed with the team on 27 October.
- Confirm the `base` columns with PM-4.
- Connect the options to the PM-2 pipeline (#15).
- Week 2: quantity and H6 comparisons, regional and age-group results, location-based validation.

## Handover

Feature options and tests ready for review by PM-2. No individual records or datasets are committed.
## Follow-up: cards 13, 15, 16 and 17 (15 November 2026)

### Objective and scope

Run the remaining #16 checks with the #17 final setup (Random Forest, balanced, 300 trees, seed 42) on the shared development folds. The holdout was not scored.

### Changed and created

- Branch `feature/16-feature-checks`, based on the #17 branch (PR #21), because the checks reuse `model_comparison.py`.
- `src/team_wtm/feature_checks.py`: ladder (card 13), H6 (card 15), group results (card 16) and region folds (card 17).
- `src/team_wtm/feature_check_charts.py`: two aggregate charts.
- `tests/test_feature_checks.py`: 4 tests for group counts, small-group flags and region folds (no model training).
- `reports/sprint2/pm3_feature_location_checks.md` and the `pm3_*` aggregate tables; charts in `reports/figures/sprint2/`.
- Individual predictions saved locally in `data/processed/` only.

### Decisions and reasoning

- Reuse PM-1's split plan and scoring and PM-4's pipeline, so every number is comparable with #17. Check: the reproduced mean fold macro F1 is 0.6828, identical to #17.
- Card 16 uses out-of-fold predictions of the final setup on the main folds. Small groups (< 30 real repairs, D20) are flagged and kept.
- Card 17 uses `GroupKFold` by region (5 folds) on development records. It differs from the main split, which only keeps identical records together.
- The quantity check (card 14) and part of the ladder were already run by PM-4 in #17 with the PM-3 options; those rows are reused, not rerun.

### Checks and results

- Ladder: base 0.616 → + age 0.631 → + missing flags 0.638 → full 0.683. Location gives the largest gain; missing flags less than one fold std.
- H6 supported: quantity alone 0.420, waterpoint type alone 0.315, full 0.683.
- Unseen regions: macro F1 0.489 ± 0.037 and repair recall 0.036, against 0.683 and 0.489 on the main folds. 163 of 3,454 repairs found.
- Age known vs unknown: macro F1 0.686 vs 0.671, but unknown age is mostly six regions, so it cannot be separated from region.
- Tests: 4 new tests pass.

### Uncertainty and remaining work

- Reasons for the regional drop are not tested (repair share differs by region; unseen region category; nearby pumps alike).
- Quantity availability at prediction time is still unconfirmed (PM-1, PM-4).
- This branch includes the #17 commits until PR #21 is merged.

### Handover

Report and charts ready for PM-4 and PM-1 for the final model decision and presentation. Review requested from PM-2.
