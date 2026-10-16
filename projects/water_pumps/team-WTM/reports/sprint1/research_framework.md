# Sprint 1 research framework

Team WTM, ReDI School Data Circle 2026. Updated 15 October 2026 for Issue #3.

This document defines the questions and hypotheses used to organise Sprint 1.
The [integration summary](README.md) records the findings.
The [decision log](decisions.md) records the choices made during analysis.

## Problem and scope

We want to understand differences in the recorded condition of Tanzanian
waterpoints and later test whether the available fields can predict that condition.

The training data contain 59,400 waterpoints and 40 input columns, including
the record ID. Labels are stored separately and give three states:

- `functional`: working;
- `functional needs repair`: working but needs repair;
- `non functional`: not working.

The records date from 2002 to 2013. They are observations of condition at the
recording date. They are not a repeated history of each waterpoint or a current
inventory. Predicting this recorded label differs from forecasting a future failure.

Sprint 1 covers data quality, descriptive analysis, candidate features and the
presentation of findings. Model training and predictive evaluation belong to
Sprint 2. A relationship in this data does not establish a cause.

## Main question

Which location, construction, water-supply and management characteristics are
associated with recorded waterpoint condition?

The later modelling question is: how well can these characteristics distinguish
the three states on records held out from training?

## Supporting questions and available fields

| Question | Fields or evidence |
| --- | --- |
| How common is each pump state? | `status_group` counts and percentages |
| Does condition vary by location? | `region`, `basin`, `lga`, coordinates |
| How does condition differ with age? | `construction_year`, `date_recorded`, derived `pump_age` |
| How does water availability relate to condition? | `quantity`, `quantity_group` |
| Do extraction technologies and outlet types differ? | `extraction_type` and its groups, `waterpoint_type` and its group |
| Do sources and recorded quality differ? | `source` and its groups, `water_quality`, `quality_group` |
| Do management and payment groups differ? | `management`, `scheme_management`, `payment_type`, related fields |
| What information is missing or uncertain? | Shared audit, missingness by state and region, source-value screens |
| Which fields help a model predict new cases? | Sprint 2 validation, including class-specific results |

## Consolidated hypotheses

These six identifiers preserve the original framework. They provide one common
reference for the different analyses. Their descriptive outcomes and limits are
in the [hypothesis results](README.md#hypothesis-results).

| ID | Hypothesis | Fields | Sprint |
| --- | --- | --- | --- |
| H1 | Older waterpoints have a lower working share than newer waterpoints. | Derived age and `status_group` | 1: descriptive comparison |
| H2 | The distribution of pump states differs by location. | Region, basin, coordinates | 1 |
| H3 | Waterpoints recorded as having enough water have a higher working share than dry or insufficient-water groups. | Quantity and state | 1 |
| H4 | Pump states differ among extraction technologies and outlet types. | Extraction and waterpoint type | 1 |
| H5 | Pump states differ among management and payment groups. | Management and payment fields | 1 |
| H6 | Combining feature groups improves prediction over a single field. | Combined predictors and held-out evaluation | 2: not yet tested |

H1 now names the working share directly. The original wording joined repair and
non-working outcomes, which could hide their different patterns. It is a
clarification of the question after inspecting the data, not a preregistered test.

The categorical notebook also discusses source, quality, installer, funder and
permit. Those are supporting analyses, not additional hypotheses silently
substituted for H1-H6. A weak one-field association for permit does not establish
that permits have no effect or cannot help a model.

The team has presented the Sprint 1 results, as confirmed by the project owner.
A separate vote or formal approval of every hypothesis has not been recorded.
Peer reviewers can confirm the consolidated framing in PR #11.

## Original data groups

The following assigns each of the 40 input columns once. The target is listed
separately. Counts describe columns, not predictive importance.

| Group | Count | Columns |
| --- | ---: | --- |
| Geography and elevation | 10 | longitude, latitude, gps_height, basin, region, region_code, district_code, lga, ward, subvillage |
| Construction, funding and installation | 3 | construction_year, funder, installer |
| Pump and outlet technology | 5 | extraction_type, extraction_type_group, extraction_type_class, waterpoint_type, waterpoint_type_group |
| Water source | 3 | source, source_type, source_class |
| Water availability and amount | 3 | quantity, quantity_group, amount_tsh |
| Water quality | 2 | water_quality, quality_group |
| Management and governance | 6 | management, management_group, scheme_management, scheme_name, permit, public_meeting |
| Payment | 2 | payment, payment_type |
| Population | 1 | population |
| Identification and recording | 4 | id, wpt_name, date_recorded, recorded_by |
| Meaning unresolved | 1 | num_private |
| Target, supplied separately | 1 | status_group |

Population describes a recorded population field, not a verified count of people
served. Funder, installer and manager describe different roles. Related category
fields can retain different levels of detail.

## Method and interpretation

1. Validate the ID join and inspect raw values before cleaning.
2. Preserve the raw files and retain rows unless there is evidence for removal.
3. Describe the target, missingness and category sizes.
4. Compare pump states within categories and locations.
5. Derive age at recording and state its coverage and exclusions.
6. Check numeric distributions and pairwise correlations, including sensitivity
   to zero heights.
7. Compare the youngest and oldest groups within regions and extraction classes.
8. Use the findings to define modelling checks, rather than claim model performance.

Cramer's V summarises a categorical field's association with the three states.
Spearman correlation compares selected numeric fields. They answer different
questions and their values should not be ranked together. No hypothesis is
declared statistically proven; these are descriptive analyses.

The available records do not provide a regular time series for each waterpoint.
Time autocorrelation is therefore not part of Sprint 1. Geographic comparisons
are descriptive and are not a formal spatial-autocorrelation analysis.

## Sprint 2 evaluation plan

Start with a majority-class baseline and a simple classifier before comparing
more complex models. Report accuracy alongside precision, recall, F1 and a
confusion matrix for each state. Test class weights rather than assuming they
will help.

Split the data before learning imputation, encoding, scaling or rare-category
groupings. Review identical feature records before assigning validation folds.
Consider location-based evaluation to test transfer to places outside training.

Compare models with and without water quantity and document whether every field
would be available at the intended prediction time. Review missingness flags,
zero heights, uncertain field meanings and related columns.

H6 remains open until those comparisons use a defined validation protocol.

## Sources

- [Data dictionary](../../data/raw/data_dictionary.md) and [quality findings](data_quality_findings.md)
- [Integrated findings and pinned analysis revisions](README.md)
- [Presentation archive and figure selection](presentations/README.md)
- [DrivenData competition description](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)
