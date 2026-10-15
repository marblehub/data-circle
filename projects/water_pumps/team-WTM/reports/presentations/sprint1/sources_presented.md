# Sprint 1 presentation: sources and artwork

Prepared on 15 October 2026. The six-slide deck is designed for about 4 minutes 30 seconds. Speaker notes contain the spoken script and handovers only. Source details are kept here.

## Repository revisions

Analysis and local cleaned export: 747b6af589f35a3bef7a50b3d70890d5512c93ae.
GitHub main at preparation: 6bc61dc0d0daa87397ca756c3afb0156c65cfca0.
Results from the current analysis branch include work not yet merged to main.

## Results used

- [Additional EDA report](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/sprint1_additional_eda.md)
- [Aggregate EDA results](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/additional_eda/summary.json)
- [Cleaned export documentation](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/cleaned_data_export.md)
- [Original feature dictionary](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/data_dictionary.md)
- [Data quality findings](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/data_quality_findings.md)

Slide 2 groups the 40 original input columns, including id. The target is supplied in the separate labels file. Every original input column is assigned once. These groups describe topics and do not measure predictive importance.

| Group | Count | Original columns |
| --- | ---: | --- |
| Location and population | 11 | longitude, latitude, gps_height, basin, region, region_code, district_code, lga, ward, subvillage, population |
| Construction and pump type | 8 | construction_year, funder, installer, extraction_type, extraction_type_group, extraction_type_class, waterpoint_type, waterpoint_type_group |
| Water source, amount and quality | 8 | source, source_type, source_class, quantity, quantity_group, amount_tsh, water_quality, quality_group |
| Management and payment | 8 | management, management_group, scheme_management, scheme_name, permit, public_meeting, payment, payment_type |
| Record details | 4 | id, wpt_name, date_recorded, recorded_by |
| Meaning still unclear | 1 | num_private |

Slide labels use Working = functional, Needs repair = functional needs repair (still working), and Not working = non functional. The same colours are used in every status chart and its accompanying text: Working #008F78, Needs repair #B77900, Not working #C6283E.

Slide 3 construction-year missing shares use each target class's full count. Unknown years are 0 placeholders under the shared audit rule. Dodoma, Kagera, Mbeya and Tabora account for 12,115 records and have no known construction years. The shared cleaned export retains all 59,400 rows and has 47 columns; num_private is dropped under the current rule, and pump_age plus six quality flags are added. Repeated observations remain for review.

Slide 4 water quantity counts are recalculated from the local original training files joined by validated ID. Enough water: working 21,648; needs repair 2,400; not working 9,138; n=33,186. Dry: working 157; needs repair 37; not working 6,052; n=6,246. Dry non-functional share is 96.894%. The age chart covers 38,682 usable ages, excluding 20,709 unknown construction years and nine negative ages.

| Age at recording | Records | Working share |
| --- | ---: | ---: |
| 0-5 | 11,630 | 71.5305% |
| 6-10 | 5,627 | 61.2760% |
| 11-20 | 9,456 | 57.6142% |
| 21-30 | 5,633 | 44.2748% |
| 31+ | 6,336 | 31.4236% |

The within-group age comparison is youngest (0-5) versus oldest (31+) with at least 30 observations in each extreme-age group. All 16 eligible regions and all five eligible extraction classes show higher working shares in the youngest group. Intermediate ages often do not form a monotonic sequence. These are cross-sectional associations.

Slide 5 uses pairwise-available Spearman correlations among pump_age (Age), population (People), amount_tsh (Amount), and gps_height (Height). The second matrix masks only zero heights. The amount-height pair changes from 0.341880 at n=59,400 to 0.022472 at n=38,962. Other pairs have different valid counts, recorded in the aggregate summary. The amount field's meaning and the interpretation of zero heights remain source-validation questions. The saved cleaned data retains the raw amount and height values. The heatmaps have a separate numeric colour scale, with purple for positive and blue for negative correlations.

Slide 6 target counts are working 32,259; needs repair 4,317; not working 22,824. Percentages use all 59,400 rows. Model comparisons and per-state evaluation are planned for Sprint 2; the deck claims no trained-model performance.

## Input checksums

- labels: Training_set_labels.csv; SHA-256 ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24.
- values: Training_set_values.csv; SHA-256 d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d.

## Speaker timing

| Slide | Speaker | Seconds | Spoken words |
| --- | --- | ---: | ---: |
| 1 | Goodfriend | 25 | 45 |
| 2 | Tayaphon | 45 | 86 |
| 3 | Morisade | 50 | 92 |
| 4 | Tari | 65 | 123 |
| 5 | Tayaphon | 45 | 97 |
| 6 | Goodfriend | 40 | 85 |

Total timing target: 270 seconds. Spoken words: 528. Actual delivery depends on pace and pauses.

## Cover artwork

Generated with the built-in image generation tool. The artwork is a conceptual illustration, not a dataset observation or documentary photograph.

Final project asset: cover_illustration.png

Final prompt:

Use case: illustration-story. Asset type: cover illustration for a school data-science presentation about waterpoint condition in Tanzania. Create one simple, elegant editorial illustration of a sturdy rural hand-operated water pump with a small stream of clean water flowing from its spout into a shallow blue pool. Show the whole pump, with generous open space around it. Flat, softly textured illustration, clean shapes, restrained detail, no people, no buildings, no landscape clutter, no graphs or data symbols. Colours: dark blue-green metal, teal water and a small warm sand-coloured ground patch. The illustration will sit on the right half of a dark blue-green slide, so use a genuinely transparent background and avoid a surrounding square or panel. Single clear subject, readable at small size. No text, no numbers, no logos, no watermark. This is a conceptual illustration, not a photograph or a data result.
