# Local cleaned training snapshot

Implemented on branch analysis/sprint1-numeric-missingness-stratified as a follow-up
to the additional Sprint 1 EDA.

## Result

The reproducible export creates these local files:

- data/processed/training_cleaned.parquet
- data/processed/training_cleaned.metadata.json

The snapshot has **59,400 rows and 47 columns**. It retains the original IDs,
target labels, and all repeated observations. The six quality flags and
derived pump age are additive. The
[processed data dictionary](../data/processed/processed_data_dictionary.md)
lists every exported column, its pandas dtype and missing count.

Both generated files are ignored by Git. The implementation, tests, dictionary
and documentation are version controlled.

## Preparation rules

The exporter imports the existing shared cleaner and applies its defaults:

- Convert zero population and construction year to missing values.
- Parse date_recorded, with unparseable values becoming NaT.
- Drop num_private under the existing feature-selection decision.

It preserves elevation and amount zeros, negative elevations, categorical
spellings and placeholders, coordinates, nonzero source construction years,
and unresolved duplicate records.

For storage, public_meeting and permit use pandas nullable Boolean dtype.
True and false retain their meaning and missing values remain missing. This
avoids an object-column null changing representation during a Parquet round-trip.

Pump age is recording year minus construction year. It is available only for
a positive construction year, a parseable recording date, and a nonnegative
calculated age. The EDA and exporter now import the same derive_pump_age
function from the shared cleaning module. Existing EDA results are unchanged.

The exporter validates unique, non-missing IDs through the existing loader.
Features and labels must have exactly the same IDs, and every target must be
one of the three known classes. Invalid or unmatched inputs stop the export.

## Derived columns and verified counts

| Column | Meaning | Flagged rows |
| --- | --- | ---: |
| construction_year_missing | Construction year is unavailable after the shared zero-placeholder rule | 20,709 |
| population_missing | Population is unavailable after the shared zero-placeholder rule | 21,381 |
| pump_age_missing | A usable age cannot be calculated | 20,718 |
| pump_age_invalid | Construction year is known but the derived age is unusable | 9 |
| coordinate_placeholder | Longitude is zero and absolute latitude is at most 1e-6 degrees | 1,812 |
| coordinate_screen_failed | Coordinates are missing/non-numeric or outside the broad audit envelope | 1,812 |

Flag counts overlap and must not be added. The nine invalid-age records in the
current data have construction years later than recording years. Usable ages
are available for 38,682 records. An age of zero is valid.

The broad audit coordinate envelope is longitude 29-41 and latitude -12.5 to
-0.9 degrees, inclusive. It is an approximate screening rectangle, not an exact
country-boundary test. Coordinates remain stored, and rows are retained.
The geospatial notebook uses a narrower latitude envelope and flags 1,819
records; the exported flag deliberately follows the documented shared audit.
Use the appropriate documented screen when preparing a particular map.

The output retains the original target counts:

| Target | Rows |
| --- | ---: |
| functional | 32,259 |
| functional needs repair | 4,317 |
| non functional | 22,824 |

The audit's 36 repeated observations remain. Different IDs may represent
different pumps, so the export does not deduplicate or change labels.

## Generate or refresh the snapshot

From projects/water_pumps/team-WTM, after placing the original training CSVs in
data/raw/:

```bash
uv sync --locked
uv run python -m team_wtm.export_cleaned
```

Regenerate an existing snapshot explicitly:

```bash
uv run python -m team_wtm.export_cleaned --overwrite
```

The module resolves its default paths from the project source location. It
also runs from notebooks/ after installing the project package. Optional
--values, --labels and --output arguments select alternate inputs and output.

An existing Parquet file or metadata file is retained unless --overwrite is
supplied. A new export is prepared in a temporary directory and read back with
an exact dataframe comparison before the files replace an existing snapshot.
The Parquet and metadata files are each replaced atomically, with metadata
installed last. They are not a single filesystem transaction; the recorded
output hash can detect an interrupted replacement that leaves a mismatched pair.
Input hashes are checked before and after loading so provenance cannot silently
describe an input that changed during the run.

The exporter uses PyArrow with Zstandard compression and writes no pandas
index column. PyArrow is now a direct project dependency pinned through uv.lock;
the existing locked package version is retained.

## Provenance metadata

The local JSON records:

- Input filenames, dimensions and SHA-256 hashes.
- UTC generation time and the cleaning rules version.
- Current Git commit, whether the two source files have uncommitted changes,
  exact hashes of those source files, and the lockfile hash.
- Python, pandas and PyArrow versions.
- Output filename, SHA-256 hash, dimensions and pandas column dtypes.
- Applied rules, missing counts, quality-flag counts, target counts and duplicate count.
- Successful exact Parquet round-trip verification.

Code hashes identify the implementation even when the working tree contains
uncommitted source changes. For a snapshot tied to a committed revision, regenerate
it after committing the source files.

## Using the snapshot

```python
import pandas as pd
from team_wtm.export_cleaned import DEFAULT_OUTPUT

training = pd.read_parquet(DEFAULT_OUTPUT)
```

The snapshot provides a reproducible starting table for EDA and modelling.
It retains categorical text, missing values, duplicate records and flagged
anomalies for subsequent decisions. Do not use id or status_group as model
predictors. Review the quality flags and source fields before selecting features.

Imputation, learned category grouping, encoding, scaling and other learned
transformations must be fitted on training folds. The export performs none of
these operations. Avoid splitting identical feature records across validation
folds without reviewing the duplicate policy.

## Validation

- Full project suite: 36 tests passed.
- Ruff lint and formatting checks passed for the changed Python files.
- The real training inputs exported successfully and round-tripped exactly,
  including nullable Boolean values and missingness.
- Input and output hashes, IDs, target counts, dimensions and quality-flag counts
  were checked.
- Git ignore checks confirmed that the Parquet and its local metadata are excluded.
- The existing EDA tests cover the shared age derivation and retain their results.

See [test_export_cleaned.py](../tests/test_export_cleaned.py) for synthetic checks
covering ID/label preservation, placeholder handling, age zero, negative ages,
coordinate screening, nullable Booleans, explicit regeneration, unmatched IDs,
invalid targets, failed writes and inputs that change during export.
