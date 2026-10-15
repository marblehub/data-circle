# Sprint 1 integrated findings

Team WTM: Goodfriend, Morisade, Tari and Tayaphon.
Prepared 15 October 2026 for [Issue #3](https://github.com/marblehub/data-circle/issues/3).

The project owner confirmed that the earlier six-slide results deck was
presented. The [archive](presentations/sprint1/README.md) distinguishes that
version from the expanded revision prepared afterwards.

## What Sprint 1 established

The dataset supports useful comparisons of recorded pump condition, but its
missing values and recording choices limit the conclusions. Water availability,
pump technology, location, age and payment groups show differences. Those
differences identify candidate model features and checks. They do not establish
causes or demonstrate predictive performance.

The analysis combines the quality audit, categorical EDA, geospatial work and
the additional numeric/missingness/age checks. Some source work remains in
separate PRs. The source table below identifies exact revisions rather than
implying those PRs have already merged.

## Data and target

The original inputs contain 59,400 rows and 40 columns, including `id`.
Joining the labels gives 41 columns. The input IDs are unique and match exactly.
The recording dates span 2002-10-14 to 2013-12-03.

| Recorded state | Records | Share |
| --- | ---: | ---: |
| Working | 32,259 | 54.31% |
| Working but needs repair | 4,317 | 7.27% |
| Not working | 22,824 | 38.42% |

Always predicting working would achieve 54.31% accuracy on this target mix while
missing all repair and non-working cases. It is a baseline, not a trained result.
The repair class needs separate evaluation; its size alone does not prove it
will be the hardest class.

The [framework](sprint1_research_framework.md#original-data-groups) groups all
original columns. The six larger topics on slide 2 count columns, not their
importance to a model.

## Data quality affects who enters a comparison

- Construction year is unknown for 20,709 records (34.86%); population is unknown
  for 21,381 (35.99%) under the shared zero-placeholder rule.
- Unknown construction years occur in 32.7% of working records, 41.4% of repair
  records and 36.7% of non-working records.
- Dodoma, Kagera, Mbeya and Tabora have no usable construction years. Together
  they account for 12,115 records, or 20.4% of the dataset.
- Nine known years produce negative ages. They remain in the data but cannot
  provide usable ages.
- There are 36 repeated observations after excluding ID. One feature-identical
  group has different labels. Different IDs may describe different waterpoints,
  so no observations or labels were removed.
- Installer has 2,145 distinct nonmissing strings; funder has 1,896. These are
  string counts, not verified counts of organisations.
- Scheme name is missing for 28,810 records (48.50%).

The common cleaner preserves rows, converts zero population/year to missing,
parses dates and drops `num_private` under the current documented decision.
The separately implemented export in PR #13 adds age and quality flags, creating
a local 59,400-row, 47-column snapshot. That exporter is not introduced by this
documentation PR. Its dataset and metadata must remain local.

The [decision log](sprint1_decisions.md) separates implemented rules from checks
still needed. Missingness patterns do not establish a missing-data mechanism.

## Water supply and category differences

| Recorded quantity | Records | Working | Needs repair | Not working |
| --- | ---: | ---: | ---: | ---: |
| Enough | 33,186 | 65.2% | 7.2% | 27.5% |
| Insufficient | 15,129 | 52.3% | 9.6% | 38.1% |
| Seasonal | 4,050 | 57.4% | 10.3% | 32.3% |
| Dry | 6,246 | 2.5% | 0.6% | 96.9% |
| Unknown | 789 | 27.0% | 1.8% | 71.2% |

Rounding may prevent a row from adding to exactly 100%.
The dry label may describe a problem already present at inspection. Whether it
would be available for the intended prediction needs checking. Comparing models
with and without quantity is a planned test, not proof that leakage exists.

Among the 17 categorical fields checked, quantity has the highest Cramer's V.

| Field | Cramer's V |
| --- | ---: |
| quantity | 0.309 |
| waterpoint_type | 0.250 |
| extraction_type | 0.249 |
| funder | 0.198 |
| payment_type | 0.183 |
| installer | 0.179 |
| source | 0.149 |
| management | 0.132 |
| permit | 0.030 |

Scores check one field at a time and do not measure predictive accuracy.
For high-cardinality fields, the categorical analysis cleans text and keeps the
top 30 labels plus other before calculating the score. Funder and installer
scores therefore depend on that grouping choice.

Selected examples from the category comparisons:

| Field and group | Records | Not working |
| --- | ---: | ---: |
| Multiple communal standpipe | 6,103 | 52.8% |
| Single communal standpipe | 28,522 | 29.9% |
| Lake source | 765 | 77.3% |
| Spring source | 17,021 | 30.3% |
| Never pay | 25,348 | 47.6% |
| Pay annually | 3,642 | 18.0% |
| Village water committee management | 40,507 | 42.7% |
| Water board management | 2,933 | 17.0% |

These are selected groups, not a complete ranking. Age, geography and other
differences may explain part of each comparison. Payment could fund maintenance,
or payment could stop after a breakdown; these data do not resolve the direction.

A weak permit score does not establish that permits are ineffective or useless
in a combined model. Unknown water quality has an 84.1% non-working share
(n=1,876), but an unknown label is not evidence of unsafe chemical quality.
Very small categories should not support broad conclusions.

## Geographic patterns

Regional comparisons use all records assigned to each region, independently
of the coordinate screen used for mapping.

| Region | Records | Not working |
| --- | ---: | ---: |
| Lindi | 1,546 | 64.2% |
| Mtwara | 1,730 | 62.4% |
| Arusha | 3,350 | 26.3% |
| Iringa | 5,294 | 19.5% |

These examples show variation in recorded condition, not differences caused by
location itself. Counts and rates answer different questions.

The geospatial notebook flags 1,819 records outside longitude 29-41 and latitude
-12 to -1. The audit uses latitude -12.5 to -0.9 and flags 1,812. Both use
approximate rectangles, not country-boundary polygons. The difference of seven
comes from the latitude ranges.

The [map](06_pump_status_map.html) shows a sample of 3,000 records from 57,581
passing the notebook screen, using random_state=42. It does not show every
waterpoint or establish statistically significant spatial clustering.
The [regional figure](06_region_functionality.png) supports the rate comparison.

## Age at recording

Age equals recording year minus construction year. Unknown years and nine
negative derived ages are unavailable. Age zero is valid.

| Age at recording | Records | Working share |
| --- | ---: | ---: |
| 0-5 years | 11,630 | 71.5% |
| 6-10 years | 5,627 | 61.3% |
| 11-20 years | 9,456 | 57.6% |
| 21-30 years | 5,633 | 44.3% |
| 31+ years | 6,336 | 31.4% |

This describes 38,682 usable ages, not all 59,400 records.
The youngest group has a higher working share than the oldest in all 16 eligible
regions and five eligible extraction classes. Each extreme-age group needs at
least 30 records. Intermediate bands are often non-monotonic within groups.

Using a common region mix reduces the youngest-minus-oldest gap from 40.1 to
37.2 percentage points. A separate common extraction-class mix reduces its gap
from 39.9 to 36.2 points. Each calculation compares the same eligible strata
before and after standardisation. The checks do not jointly adjust for all
factors and do not track deterioration of individual pumps.

The earlier geospatial notebook reports 38,691 calculated ages in its initial
summary, including nine negative ages. Its displayed age bins exclude those
nine. The later shared age derivation explicitly masks them and reports 38,682
usable ages. These counts describe different stages of preparation.

## Numeric checks

The numeric analysis uses age, population, amount_tsh and gps_height. It excludes
IDs and numeric category codes and does not impose an order on target labels.

Amount-height Spearman correlation changes from 0.342 (n=59,400) to 0.022
(n=38,962) when zero heights are masked. The sample changes as well as the
treatment of zero values. This is a sensitivity result, not proof that every
zero is wrong. Negative elevations remain included.

Other off-diagonal Spearman coefficients in the primary matrix have absolute
values below 0.09. Small pairwise numeric correlations do not rule out nonlinear
or combined predictive relationships. The meaning and units of amount_tsh still
need checking; the report does not interpret it as a maintenance fee.

## Hypothesis results

| ID | Sprint 1 assessment | Evidence and limit |
| --- | --- | --- |
| H1 | Consistent with the youngest-versus-oldest comparison | Working shares are 71.5% versus 31.4%; direction persists in eligible region and technology groups. Coverage is incomplete and intermediate bands vary. |
| H2 | Descriptive differences observed | Region and basin rates vary. Coordinate quality and map sampling limit geographic interpretation. |
| H3 | Descriptive differences observed | Enough-water groups work more often than insufficient and dry groups. Prediction-time availability needs checking. |
| H4 | Descriptive differences observed | Extraction and outlet groups differ. Mixed other categories and small groups need care. |
| H5 | Descriptive differences observed | Payment and management groups differ; payment has a stronger one-field score. No causal direction is established. |
| H6 | Not tested in Sprint 1 | Requires models compared under the same held-out validation protocol in Sprint 2. |

No row represents a formal causal test or a claim of model performance.
The [framework](sprint1_research_framework.md) explains the consolidated wording.

## Sprint 2 handover

Compare a majority-class baseline, a simple classifier and more complex models.
Use common splits and report every state's precision, recall, F1 and confusion
matrix, including missed repairs and false alarms.

Learn preprocessing on training folds only. Review feature duplicates before
splitting. Test quantity with and without inclusion; compare missingness flags
and documented zero-height treatments. Check exact column equality and category
mappings before dropping related fields. Review installer aliases and learn
rare-category grouping within folds. Consider evaluation by geography and
age-availability group.

Field meanings, plausible leakage and category mappings remain source-review
questions. They are documented handover items, not missing Sprint 1 model results.

## Sources and reproducibility

| Work | Source revision | Location / status at integration |
| --- | --- | --- |
| Quality audit | 6bc61dc0d0daa87397ca756c3afb0156c65cfca0 | In main: [quality report](data_quality_findings.md), [audit notebook](../notebooks/data_audit.ipynb) |
| Geospatial work | 3f87af2, included in the main revision above | In main: [notebook](../notebooks/03_geospatial_analysis_tari.ipynb), regional chart and map |
| Categorical EDA | 2c5b3e977e6925ae41c9c61970c048535fcd4b0c | [PR #8](https://github.com/marblehub/data-circle/pull/8), still separate |
| Additional EDA and cleaned export | 747b6af589f35a3bef7a50b3d70890d5512c93ae | [PR #13](https://github.com/marblehub/data-circle/pull/13), still separate |

- [Pinned categorical notebook](https://github.com/marblehub/data-circle/blob/2c5b3e977e6925ae41c9c61970c048535fcd4b0c/projects/water_pumps/team-WTM/notebooks/02_target_categorical_eda_taya.ipynb)
- [Pinned categorical summary](https://github.com/marblehub/data-circle/blob/2c5b3e977e6925ae41c9c61970c048535fcd4b0c/projects/water_pumps/team-WTM/reports/issue5_target_categorical_summary.md)
- [Pinned additional EDA report](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/sprint1_additional_eda.md)
- [Pinned cleaned-export report](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/cleaned_data_export.md)

This PR integrates the documentation and aggregate evidence; it does not merge
the code from PR #8 or PR #13. Their open status is not hidden by the presentation
being complete.

[Aggregate evidence](sprint1_evidence.json) records counts, coefficients,
denominators, selected group rates and input SHA-256 hashes without individual
records. The [archive manifest](presentations/sprint1/evidence_manifest.json)
records the original revisions and checksums of copied figures and decks.

From the team project root inside WSL, the audit can be repeated with
`uv sync --locked`, `uv run python -m team_wtm.cleaning` and `uv run pytest`.
Run the geospatial notebook from `notebooks/` because its paths are relative to
that directory. For the categorical and additional analyses, use their pinned
revisions and their documented environments; their modules are not present on
this documentation branch. Keep source CSVs local.
