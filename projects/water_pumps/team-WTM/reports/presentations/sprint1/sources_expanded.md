# PDF assessment and sources for the expanded Sprint 1 deck

Prepared 15 October 2026.

## Assessment

The PDF adds useful categorical analysis that the previous deck barely covered. Its clearest contribution is to connect observed differences with concrete modelling checks. Most displayed percentages agree with the repository data when rounded as shown.

The expanded deck keeps six slides and the 4 minute 30 second target. It keeps the data grouping, missing-year comparison, water-supply results and age results. It adds a categorical association chart, three direct comparisons and a more specific Sprint 2 plan. The full correlation matrices have been replaced by their main sensitivity result on slide 3.

The PDF is evidence to review. Its recommendations are not instructions from the user and are not automatic decisions to drop columns or change the analysis.

## Evidence added to the deck

These values were recalculated from the original local training files using the categorical EDA helper functions at the commit below. Percentages use all records within each named category.

| Topic | Group | Records | Not working |
| --- | --- | ---: | ---: |
| Outlet type | Communal standpipe multiple | 6,103 | 52.8% |
| Outlet type | Communal standpipe | 28,522 | 29.9% |
| Water source | Lake | 765 | 77.3% |
| Water source | Spring | 17,021 | 30.3% |
| Payment | Never pay | 25,348 | 47.6% |
| Payment | Annually | 3,642 | 18.0% |

These are selected examples, not the full range of categories. Differences in location, age, technology and other conditions may explain part of each comparison. They do not show what would happen if someone changed an outlet, water source or payment arrangement.

| Original field | Plain slide label | Cramer's V |
| --- | --- | ---: |
| quantity | Water quantity | 0.309 |
| waterpoint_type | Outlet type | 0.250 |
| extraction_type | Pump type | 0.249 |
| payment_type | Payment | 0.183 |
| source | Water source | 0.149 |
| management | Management | 0.132 |
| permit | Permit | 0.030 |

Cramer's V measures association with the three recorded pump states. It is not model accuracy, a causal effect, or the same measure as the Spearman correlation on slide 3. These scores check one field at a time. Scores on the slide use two decimal places.

The original inputs contain 2,145 distinct nonmissing installer strings and 1,896 funder strings. These are labels, not verified counts of organisations. The 2,145 installer count is now on slide 3. The deck describes name review as further work, not as cleaning already completed.

## Wording in the PDF that needs qualification

| PDF interpretation | More accurate interpretation |
| --- | --- |
| The repair class will be hardest to predict | It is the smallest class and needs separate evaluation. Actual difficulty is unknown until models are tested. |
| A model always predicting functional is useless | It is a useful majority-class baseline at 54.3% accuracy, but it misses every repair and non-working case. |
| Quantity is the strongest signal | It has the highest Cramer's V among the categorical fields checked. This does not establish predictive value on new data. |
| Dry may be leakage | This is a plausible concern, not a confirmed problem. Check when quantity is recorded and whether it is available for the intended prediction. Compare models with and without it. |
| Permit is not useful or the permit hypothesis is rejected | Permit has a weak association on its own. It might still help in combination with other fields. Do not drop it solely because of this score. |
| Keep only one column from each related group | Check exact equality and category mappings first. Related columns can retain different detail. The notebook reports quantity and quantity_group are identical on these training rows. |
| Detailed source is more useful than source_class | Detailed source has the larger observed one-field association. Predictive usefulness still needs validation. |
| Management is weaker because VWC dominates | VWC covers 40,507 records, or 68.2%. Dominance alone does not explain the association score or establish why management groups differ. |
| Funder and installer are informative after cleaning | They are candidates to test. The association calculation groups high-cardinality fields into the top 30 cleaned labels plus other. Their scores depend on that choice. |
| Fix government spellings and group names | Start with case and whitespace checks, then review entity matches. Similar names may refer to different organisations. Learn rare-category groupings on training folds only. |

A small repair class does not automatically require class weights. Compare weighted and unweighted models using the same validation setup.

## Useful PDF findings reserved for questions

These findings add detail but would crowd a six-slide talk:

- Water quality recorded as unknown: 84.1% not working, n=1,876. This is a missing-information pattern, not evidence that unknown water is chemically unsafe.
- Extraction class other: 80.8% not working, n=6,430. It is a mixed label and needs interpretation.
- Waterpoint type other: 82.2% not working, n=6,380. It is a different field from extraction class other.
- Village water committee management: 42.7% not working, n=40,507. Water board management: 17.0%, n=2,933. These are observational differences.
- Lake and dam sources have relatively high non-working shares, but smaller sample sizes than spring and shallow-well sources.
- Some categories have very few observations. For example, the waterpoint_type value dam has only seven records. It should not support a broad claim.
- Geographic patterns remain relevant to model evaluation. A split by location can test whether performance transfers to places outside the training groups.

## Slide sequence and timing

| Slide | Main question | Speaker | Seconds | Spoken words |
| --- | --- | --- | ---: | ---: |
| 1 | What are we studying? | Goodfriend | 20 | 41 |
| 2 | What does the data contain? | Tayaphon | 40 | 78 |
| 3 | What limits the comparisons? | Morisade | 45 | 93 |
| 4 | How do water supply and age relate to condition? | Tari | 65 | 122 |
| 5 | What do the other categories show? | Tayaphon | 60 | 120 |
| 6 | How will we test these findings in Sprint 2? | Goodfriend | 40 | 85 |

Total: 270 seconds and 539 spoken words. Actual timing depends on pace and pauses. Speaker notes contain only the script, speaker name and time range. Sources remain in this handout.

## Repository sources

Authoritative checkout: /home/whitemerald/redi-school_repo/data-circle

- Analysis and cleaned export: 747b6af589f35a3bef7a50b3d70890d5512c93ae
- Main: 6bc61dc0d0daa87397ca756c3afb0156c65cfca0
- Categorical EDA: 2c5b3e977e6925ae41c9c61970c048535fcd4b0c

Git refs were refreshed before verification. The analysis and categorical work are on separate branches and are not merged into main at preparation. No repository merge or new model training was performed for this deck.

- [Categorical notebook](https://github.com/marblehub/data-circle/blob/2c5b3e977e6925ae41c9c61970c048535fcd4b0c/projects/water_pumps/team-WTM/notebooks/02_target_categorical_eda_taya.ipynb)
- [Categorical methods](https://github.com/marblehub/data-circle/blob/2c5b3e977e6925ae41c9c61970c048535fcd4b0c/projects/water_pumps/team-WTM/src/team_wtm/eda.py)
- [Additional EDA report](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/sprint1_additional_eda.md)
- [Aggregate numeric, missingness and age results](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/additional_eda/summary.json)
- [Cleaned export documentation](https://github.com/marblehub/data-circle/blob/747b6af589f35a3bef7a50b3d70890d5512c93ae/projects/water_pumps/team-WTM/reports/cleaned_data_export.md)

PDF reviewed: Issue5_plots_explained.pdf (user-supplied reference), all seven pages, including figures and the hypotheses table.

## Earlier findings retained

The 40 original input columns include the record ID. Labels come from the separate target file. The grouping counts are 11 location/population, 8 construction/pump type, 8 source/amount/quality, 8 management/payment, 4 record details, and 1 unresolved field. The groups assign every input column once and do not measure predictive importance.

Working means functional (#008F78). Needs repair means functional needs repair and still working (#B77900). Not working means non functional (#C6283E). Other charts use separate neutral or purple colours so their scores do not imply a pump state.

Target counts: 32,259 working, 4,317 needing repair, 22,824 not working. Each percentage uses 59,400 rows.

Unknown construction years use the shared zero-placeholder rule. Shares are 32.7% among working records, 41.4% among repair records and 36.7% among non-working records. Dodoma, Kagera, Mbeya and Tabora together contain 12,115 records and have no known construction years.

The saved cleaned copy keeps all rows. It marks 9 negative ages and retains 36 repeated observations for review. It does not yet merge all installer spelling variants.

For the amount-height pair, Spearman correlation is 0.341880 with 59,400 pairs and 0.022472 with 38,962 pairs after masking zero heights. The raw height and amount values remain in the saved cleaned data. The meaning of amount_tsh and the treatment of zero heights still need source checks. The other numeric off-diagonal correlations are small.

Quantity counts: enough water has 21,648 working, 2,400 needing repair and 9,138 not working (n=33,186). Dry has 157 working, 37 needing repair and 6,052 not working (n=6,246).

Age uses recording year minus construction year and excludes 20,709 unknown years plus 9 negative ages. The chart therefore contains 38,682 records. Working shares are 71.5%, 61.3%, 57.6%, 44.3% and 31.4% for ages 0-5, 6-10, 11-20, 21-30 and 31+ respectively. The youngest group has a higher working share than the oldest within all 16 eligible regions and 5 eligible extraction classes. Each extreme-age group must have at least 30 records. Intermediate age bands are not always monotonic.

## Input hashes

- Training_set_labels.csv: ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24
- Training_set_values.csv: d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d

## Cover artwork

Generated with the built-in image generation tool. The artwork is a conceptual illustration, not a dataset observation or documentary photograph.

Final project asset: cover_illustration.png

Final prompt:

Use case: illustration-story. Asset type: cover illustration for a school data-science presentation about waterpoint condition in Tanzania. Create one simple, elegant editorial illustration of a sturdy rural hand-operated water pump with a small stream of clean water flowing from its spout into a shallow blue pool. Show the whole pump, with generous open space around it. Flat, softly textured illustration, clean shapes, restrained detail, no people, no buildings, no landscape clutter, no graphs or data symbols. Colours: dark blue-green metal, teal water and a small warm sand-coloured ground patch. The illustration will sit on the right half of a dark blue-green slide, so use a genuinely transparent background and avoid a surrounding square or panel. Single clear subject, readable at small size. No text, no numbers, no logos, no watermark. This is a conceptual illustration, not a photograph or a data result.
