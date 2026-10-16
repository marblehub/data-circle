# Sprint 1 analytical decision log

Prepared 15 October 2026 for Issue #3. This records the current rules and open
questions.

## Data preparation

| ID | Decision and reason | Status / location |
| --- | --- | --- |
| D01 | Preserve the original CSVs. Validate unique, nonmissing IDs and inspect join coverage before analysis. | Implemented in the audit and shared cleaner in main. |
| D02 | Treat zero population and construction year as unknown under the project convention. | Implemented in the shared cleaner. Counts: 21,381 and 20,709. |
| D03 | Parse recording dates. Derive age from the recording year, not today's year. | Date parsing is in main; shared age derivation is in PR #13. |
| D04 | A missing/nonpositive year, unusable date or negative derived age cannot supply a usable age. Preserve nonzero source years while reviewing errors. | Explicitly implemented in PR #13. Nine known ages are negative; 38,682 are usable. |
| D05 | Keep all observations and labels, including repeated feature records. Different IDs do not prove the same waterpoint. | Implemented. Thirty-six repeated observations and one feature-identical group with conflicting labels need source review. |
| D06 | Drop num_private by default because its meaning is unresolved. Do not call it constant or invalid. | Existing cleaner decision. It has 65 distinct values; callers can retain it with drop_columns=(). |
| D07 | Keep uncertain amount and elevation values, including zeros and negative elevations. Do not silently clip large values. | Implemented in the common cleaner. Source interpretation remains open. |
| D08 | Retain raw categorical strings. Case/whitespace variants are candidate matches, not confirmed entities. | Audit reports variants. Broader name consolidation is deferred. |
| D09 | Retain related categorical columns until equality or mappings are checked. | Current common cleaner. Categorical EDA finds quantity equals quantity_group on all training rows; this does not establish the same rule for every related pair. |
| D10 | Keep a separate cleaned snapshot with provenance rather than replacing raw files. | Export implemented in separate PR #13: 59,400 rows, 47 columns, exact Parquet round-trip, input/code/output hashes. Generated data and metadata remain local. |

## Coordinates and maps

| ID | Decision and reason | Status / location |
| --- | --- | --- |
| D11 | Preserve coordinates in shared data, but exclude unsuitable locations from maps and distance calculations. Screens are approximate rectangles, not verified country boundaries. | Audit uses longitude 29-41, latitude -12.5 to -0.9: 1,812 failures. Geospatial notebook uses longitude 29-41, latitude -12 to -1: 1,819 failures. Seven additional points explain the count difference. |
| D12 | Distinguish a map sample from a complete inventory. | The existing map samples 3,000 of 57,581 passing the geospatial screen, random_state=42. Regional rates use the full regional records. |
| D13 | Treat geographic patterns as descriptive. | No formal spatial-autocorrelation result or causal effect of location is claimed. A spatial-dependence check is a possible later analysis. |

## Comparisons and interpretation

| ID | Decision and reason | Status / location |
| --- | --- | --- |
| D14 | Compare pump states within categories using all records in the named category as the denominator. Show sample sizes. | Categorical EDA in PR #8. Missing and textual unknown labels remain distinct where specified. |
| D15 | Use Cramer's V for one-field categorical association, not model accuracy. | PR #8 checks 17 fields. High-cardinality fields use cleaned top-30 labels plus other. Rankings depend on this preprocessing. |
| D16 | Do not drop permit or another field solely because its one-field association is weak. | Modelling decision deferred to validation. No causal conclusion about permits. |
| D17 | Use only age, population, amount_tsh and gps_height in the numeric matrices. Exclude IDs, numeric category codes and numeric encodings of target labels. | PR #13. Spearman is primary; Pearson and pairwise counts are also recorded. |
| D18 | Keep zero heights in the primary correlation and mask them only in a named sensitivity check. | PR #13. Amount-height Spearman changes from 0.342 (n=59,400) to 0.022 (n=38,962). Both value treatment and observed sample change. |
| D19 | Compare age groups 0-5, 6-10, 11-20, 21-30 and 31+. Age zero is valid. | Existing geospatial plot and later shared derivation. Reports distinguish calculated ages including errors from usable ages. |
| D20 | Show age coverage and small cells. Require at least 30 records in both extreme-age groups for within-group contrasts. | PR #13. All regions/classes remain in tables; small rates are hidden in heatmaps. The threshold does not guarantee precise estimates. |
| D21 | Compare youngest and oldest within regions and extraction classes, and use a common mix for each separate standardisation. | PR #13. Same strata and weights are used for the two age groups. Region and technology are not jointly adjusted. No ageing effect is estimated. |
| D22 | Report patterns as associations and preserve contrary evidence. | H1-H5 have descriptive evidence with limits. Intermediate age bands often vary. H6 awaits Sprint 2 model comparisons. |

## Presentation and model handover

| ID | Decision and reason | Status / location |
| --- | --- | --- |
| D23 | Keep six slides and distinguish prepared timing from measured delivery. | The presented file has embedded notes totalling approximately five minutes. Actual elapsed delivery time was not recorded here. |
| D24 | Keep supporting figures and source references alongside the presentation. | The presentation README links the retained figures to their source analysis. The integration summary records the analysis revisions and input checksums. |
| D25 | Preserve which version was presented. | The owner identifies wtm_sprint1.pptx as presented. This is the retained presentation file. |
| D26 | Test quantity with and without inclusion and establish prediction-time availability. | Sprint 2 task. Possible leakage is a concern, not a confirmed diagnosis. |
| D27 | Split before learning imputation, scaling, encodings or category groupings. Review identical feature records when assigning folds. | Sprint 2 task; the cleaned export performs no learned preprocessing. |
| D28 | Compare a majority baseline and trained models with results for every state. | Sprint 2 task: precision, recall, F1, confusion matrices, missed repairs and false alarms. Class weights are an option to test. |
| D29 | Avoid time autocorrelation without repeated, ordered measurements of the same units. | Not part of Sprint 1; these waterpoint records are not a regular pump-level time series. |

## Open questions carried forward

- Confirm amount_tsh's meaning and units and which zero heights are placeholders.
- Review source dates/years for the nine negative ages and labels in the conflicting group.
- Review installer/funder aliases without joining distinct organisations.
- Check which inspection fields exist at the intended prediction time.
- Compare geographically separated validation and results by age-availability group.
- Validate choices about missing indicators, related columns, skewed values and class weights.

These are defined handover questions. Sprint 1 does not require a trained model
or a claimed causal explanation to close.

## Evidence

See the [integration summary](README.md), [quality report](data_quality_findings.md),
[presentation archive](presentations/README.md).
The integration summary pins the source revisions for work not yet in main.
