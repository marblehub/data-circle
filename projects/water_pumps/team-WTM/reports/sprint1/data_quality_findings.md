# Data Quality Findings

This audit supports [issue #4](https://github.com/marblehub/data-circle/issues/4).
Counts were verified against the local training CSVs on 14 October 2026.
The [audit notebook](../../notebooks/data_audit.ipynb) shows the checks and imports
shared logic from [cleaning.py](../../src/team_wtm/cleaning.py). The
[data dictionary](../../data/raw/data_dictionary.md) describes every column and the target.

## Dataset Structure and Target

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| `Training_set_values.csv` | 59,400 | 40 |
| `Training_set_labels.csv` | 59,400 | 2 |
| Merged training data | 59,400 | 41 |

Each row represents a waterpoint. `id` links the features and labels. Both input
files have 59,400 unique, non-missing IDs; the outer join matches all
59,400 records with no unmatched IDs. The 40 input columns include
one identifier and 39 potential predictors. Ten raw columns have numerical
types, including the identifier and geographic codes. The remaining columns
contain text or boolean values; `date_recorded` is initially text.

| Target class | Rows | Percent of rows |
| --- | ---: | ---: |
| `functional` | 32,259 | 54.31% |
| `functional needs repair` | 4,317 | 7.27% |
| `non functional` | 22,824 | 38.42% |

There are no missing or unexpected target labels. The repair class is smaller
than the other classes; use class-specific metrics during modelling.

## Missing Values

These are actual null counts before cleaning. Zero and string placeholders are
reported separately. Percentages use all 59,400 rows.

| Column | Missing rows | Percent of rows |
| --- | --- | --- |
| scheme_name | 28,810 | 48.50% |
| scheme_management | 3,878 | 6.53% |
| installer | 3,655 | 6.15% |
| funder | 3,637 | 6.12% |
| public_meeting | 3,334 | 5.61% |
| permit | 3,056 | 5.14% |
| subvillage | 371 | 0.62% |
| wpt_name | 2 | 0.00% |

All other columns have zero raw nulls. In particular, zero numeric nulls does
not establish complete data: population and construction year contain unknown
values encoded as zero. The default cleaner introduces 21,381 population nulls
and 20,709 construction-year nulls by replacing those zeros. Missing target
labels are never imputed. Feature imputation is deferred to a preprocessing
pipeline fitted only on training folds; `scheme_name` needs particular care
because nearly half its values are missing.

## Duplicate Observations

| Measure | Count |
| --- | --- |
| Repeated observations excluding id | 36 |
| Repeated feature rows excluding id and target | 37 |
| Feature duplicate groups | 35 |
| Rows in feature duplicate groups | 72 |
| Feature duplicate groups with conflicting labels | 1 |

Repeated-row counts exclude the first occurrence; rows in duplicate groups
include all members. Missing features are included in these comparisons.
The first comparison includes the target; the second excludes it and finds
one group with identical features but different labels. IDs themselves are
unique, so matching feature records do not prove that the underlying pumps
are the same.

Retain all records in this audit. Review the conflicting-label group with the
source before changing labels or removing rows. During modelling, consider
keeping identical feature records in the same validation fold to reduce leakage.
The notebook displays group sizes and label-conflict flags without exporting
raw observations.

## Suspicious and Invalid Values

The checks below are screening rules. Flags can overlap; their counts cannot
be added to obtain a total number of invalid records.

| Check | Rows | Percent of rows |
| --- | --- | --- |
| population: non-numeric value | 0 | 0.00% |
| construction_year: non-numeric value | 0 | 0.00% |
| gps_height: non-numeric value | 0 | 0.00% |
| longitude: non-numeric value | 0 | 0.00% |
| latitude: non-numeric value | 0 | 0.00% |
| population: zero | 21,381 | 35.99% |
| population: negative | 0 | 0.00% |
| construction_year: zero | 20,709 | 34.86% |
| construction_year: negative | 0 | 0.00% |
| gps_height: zero | 20,438 | 34.41% |
| gps_height: negative | 1,496 | 2.52% |
| construction_year: positive year before 1900 | 0 | 0.00% |
| construction_year: fractional year | 0 | 0.00% |
| date_recorded: unparseable non-missing value | 0 | 0.00% |
| construction_year: after recording year | 9 | 0.02% |
| coordinates: longitude zero and latitude near zero | 1,812 | 3.05% |
| coordinates: outside global bounds | 0 | 0.00% |
| coordinates: outside broad Tanzania envelope | 1,812 | 3.05% |
| funder: literal zero | 777 | 1.31% |
| funder: blank or unknown token | 4 | 0.01% |
| installer: literal zero | 777 | 1.31% |
| installer: blank or unknown token | 4 | 0.01% |
| status_group: unexpected non-missing label | 0 | 0.00% |

- **Population:** 21,381 zeros follow the project's unknown-value convention;
  replace them with missing values. There are no negative values.
- **Construction year:** 20,709 zeros are unknown years. Nonzero years range
  from 1960 to 2013. No positive years are before the screening threshold of
  1900 and no years are fractional. Nine records have a construction year
  after their recording year, as shown below. Preserve these nonzero years
  pending source review rather than silently correcting them.
- **Coordinates:** 1,812 records pair zero longitude with latitude within
  `1e-6` degrees of zero. These are unsuitable as Tanzanian pump locations.
  The broad screening envelope uses longitude 29–41 and latitude -12.5–-0.9
  degrees; it is not an exact country-boundary check. All records outside that
  envelope are the zero-coordinate placeholders, and none violate global
  coordinate bounds. Retain the raw coordinates in the shared cleaner, exclude
  placeholder locations from maps and distance calculations, and consider
  representing them as missing in later geospatial preprocessing.
- **Elevation:** 20,438 zeros (34.41%) may indicate unrecorded elevation; they
  do not prove that pumps are at sea level. The 1,496 negative elevations can
  be physically valid. Preserve both while seeking source validation. Consider
  a missing-elevation indicator or conversion of confirmed placeholders later.
- **Funder and installer:** each contains 777 literal `"0"` values, in the same
  777 records, plus four blank or unknown tokens. These are suspected unknown
  categories, separate from nulls. Preserve them for this audit; use a reviewed
  mapping to an unknown category or missing value during preprocessing.
- **Dates:** the recording dates range from 2002-10-14 to
  2013-12-03; no non-missing dates fail parsing. The cleaner parses
  dates and converts any future unparseable inputs to `NaT`.
- **`num_private`:** has 65 distinct values and 58,643 zeros, so it is not
  constant. Its meaning remains unclear. Dropping it is the current feature
  selection decision, not evidence that its values are invalid or useless.
  Set `drop_columns=()` to retain it for further investigation.

Construction-year inconsistencies:

| Construction year | Recording year | Rows |
| --- | --- | --- |
| 2009 | 2004 | 3 |
| 2008 | 2004 | 2 |
| 2007 | 2004 | 1 |
| 2005 | 2004 | 1 |
| 2011 | 2004 | 1 |
| 2006 | 2004 | 1 |

## Categorical Cardinality and Inconsistent Spellings

For this summary, high cardinality means at least 100 unique non-missing values.
The notebook reports every categorical column, with missing values counted
separately. `date_recorded` is excluded because it represents a date.

| Column | Unique Values | Missing Values |
| --- | --- | --- |
| wpt_name | 37,399 | 2 |
| subvillage | 19,287 | 371 |
| scheme_name | 2,695 | 28,810 |
| installer | 2,145 | 3,655 |
| ward | 2,092 | 0 |
| funder | 1,896 | 3,637 |
| lga | 125 | 0 |

Names and installer/funder fields may need rare-category handling or other
encoding; fit any learned grouping within training folds. Geographic fields
may remain useful despite having many categories. `region` has 21 unique
values and does not meet this summary's high-cardinality threshold.

The case-and-whitespace comparison identifies candidate variant groups:

| Column | Groups | Rows |
| --- | --- | --- |
| installer | 178 | 36,479 |
| scheme_name | 153 | 4,910 |
| wpt_name | 1 | 2 |

For example, installer spellings `AMREF` and `Amref` occur in a combined
443 rows. Normalize case and whitespace only through a reviewed mapping;
matching text does not by itself confirm identical entities. Abbreviations
and spelling differences remain source-review questions. No funder variant
groups are found by this narrow rule; that does not rule out semantic aliases.

Related fields describe different roles or levels of detail:

| Feature | Related feature | Decision |
| --- | --- | --- |
| `funder` | `installer` | Funding and installation are different roles; retain both. |
| `source` | `source_type` | Retain both pending a comparison of their mappings. |
| `quantity` | `quantity_group` | Retain both pending a comparison of their mappings. |
| `water_quality` | `quality_group` | Retain both pending a comparison of their mappings. |
| `extraction_type` | `extraction_type_group` / `extraction_type_class` | Retain the different levels pending investigation. |
| `waterpoint_type` | `waterpoint_type_group` | Retain both pending investigation. |
| `management` | `management_group` | Retain both pending investigation. |
| `payment` | `payment_type` | Retain both pending a comparison of their mappings. |

Related meaning is insufficient evidence for dropping a column. Feature
selection and category consolidation are separate modelling decisions.

## Cleaning Decisions

| Problem | Current shared cleaner | Further treatment or investigation |
| --- | --- | --- |
| Zero population and construction year | Replace zeros with missing values. | Consider missing indicators and imputation within training folds. |
| Recording-date text | Parse dates; invalid inputs become `NaT`. | Review any dates that fail parsing. |
| Unclear `num_private` meaning | Drop by default; allow retention via `drop_columns=()`. | Verify meaning and assess usefulness before permanent exclusion. |
| Duplicate observations and label conflicts | Preserve rows and labels. | Verify with the source; consider grouping feature duplicates for validation. |
| Nonzero construction years after recording | Preserve values and report anomalies. | Confirm which date or year is incorrect. |
| Zero coordinate placeholders | Preserve values and report counts. | Exclude from geospatial analysis; consider missing-value conversion. |
| Frequent zero and negative elevations | Preserve values and report counts. | Confirm placeholder meanings; do not assume zero means sea level. |
| Funder/installer placeholders and text variants | Preserve raw categories and report counts. | Build reviewed placeholder and alias mappings. |
| Missing features and related categorical columns | Preserve them. | Evaluate imputation and feature selection during modelling. |

The notebook preserves `train_data` and creates a separate `cleaned_data`.
Default cleaning retains 59,400 rows and produces 40 columns after
removing `num_private`. Rows, IDs and labels are not removed or imputed. The
script and notebook return data in memory and do not save a processed dataset.

## Reproducing the Audit

From the project root, install the locked environment with `uv sync`. Obtain
the training data from the [official competition source](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)
and save it locally with these exact filenames:

```text
data/raw/Training_set_values.csv
data/raw/Training_set_labels.csv
```

Run the notebook from a fresh kernel, the shared pipeline, and the checks:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=300 notebooks/data_audit.ipynb
uv run python -m team_wtm.cleaning
uv run pytest
uv run ruff check src/team_wtm/cleaning.py tests notebooks/data_audit.ipynb
```

The notebook can also run interactively with the project root or `notebooks/`
as its working directory. Its comparisons use deterministic screening rules
and do not depend on the current date. If inputs change, rerun all cells and
refresh this report's counts. Tests cover label conflicts, missing duplicate
features, screening boundaries, malformed values, categorical variants and the
shared cleaning workflow.

Validation on 14 October 2026: all 28 notebook code cells executed in order
in a fresh Jupyter kernel against the original training CSVs, with no error
outputs. The standalone pipeline completed with 59,400 rows and 40 cleaned
columns. All 18 regression tests and the Ruff checks passed.

CSV datasets are ignored by `.gitignore` and remain local. Only documentation,
code, tests and notebook summaries are version controlled; no raw dataset file
is committed.

## Acceptance Criteria Evidence

| Issue #4 criterion | Evidence |
| --- | --- |
| Raw structure documented | Dataset structure section; notebook dimensions, types and merge coverage. |
| Useful feature information | `data_dictionary.md` lists all 41 columns and the three target classes. |
| Missing values quantified | Raw null table; notebook compares raw and cleaned missing counts. |
| Duplicates investigated | Complete-observation and feature comparisons, group sizes and label conflicts. |
| Important suspicious values documented | Screening counts, construction-year inconsistencies and treatment decisions. |
| High-cardinality variables identified | Full notebook cardinality table and the threshold-based summary above. |
| Proposed cleaning decisions documented | Cleaning decisions table, notebook and shared cleaner. |
| Analysis reproducible | Fresh-kernel execution instructions, shared functions and regression tests. |
| No raw dataset committed | Ignored local CSVs; no raw dataset files in the changes. |
