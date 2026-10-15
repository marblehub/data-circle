# Sprint 1 presentation archive

Team WTM, ReDI School Data Circle 2026.

The project owner confirmed that the earlier six-slide results deck was
presented. Keep that version as the record of the presentation. The expanded
version was prepared later after reviewing the categorical EDA explanation.

| Version | File | Status | Prepared script |
| --- | --- | --- | --- |
| Presented six-slide version | [PowerPoint](wtm_sprint1_presented_2026-10-15.pptx) | Identified as presented by the owner | 528 words, 4:30 target |
| Expanded six-slide revision | [PowerPoint](wtm_sprint1_expanded_2026-10-15.pptx) | Later revision, not the presented version | 539 words, 4:30 target |

15 October 2026 is the deck preparation/archive date, not a claim about the exact
presentation session date. The actual elapsed presentation time and a verbatim
transcript were not supplied. Prepared notes and timing are preserved; do not
report them as a measured recording.

- [Prepared notes for the presented deck](speaker_notes_presented.md)
- [Prepared notes for the expanded deck](speaker_notes_expanded.md)
- [Sources and full cover-art prompt for the presented deck](sources_presented.md)
- [PDF assessment and sources for the expanded revision](sources_expanded.md)
- [File checksums and figure provenance](evidence_manifest.json)
- [Cover illustration](cover_illustration.png)

## Presented story and figure selection

The cover introduces the project. Slides 2-6 explain the data, its limits,
the descriptive findings and the plan for modelling.

| Slide | Main point | Evidence selected | Why it belongs |
| --- | --- | --- | --- |
| 1 | Water pumps in Tanzania, Team WTM | Generated conceptual hand-pump illustration | Introduces the subject without adding a data claim. |
| 2 | Group the columns to guide the questions | Editable column-count chart: 11/8/8/8/4/1; 59,400 records; three states | Establishes what the data can describe. Column counts are not feature importance. |
| 3 | Missing years limit age comparisons | Editable missing-year bars: 32.7/41.4/36.7%; four regions with 12,115 records | States the coverage limit before showing age results. |
| 4 | Water supply and age relate to condition | Editable quantity shares and age working-share bars; usable-age n=38,682 | Shows the main descriptive comparisons and within-group checks. |
| 5 | Zero heights change one numeric result | Two editable Spearman tables; amount-height 0.34 versus 0.02 | Explains why uncertain source values need sensitivity checks. |
| 6 | Evaluate every pump state in Sprint 2 | Editable target-share bars and next steps; 4,317 repair cases | Connects the small repair class to class-specific evaluation. |

All states use Working #008F78, Needs repair #B77900 and Not working #C6283E.
Needs repair means still working. Numeric correlation colours have a separate
scale. Slide text and its matching chart use the same meaning for each colour.

## Changes in the later expanded revision

The expanded deck adds target percentages to slide 2 and installer-name and
correlation checks to slide 3. It makes the possible prediction-time issue with
quantity explicit on slide 4.

Slide 5 replaces the full numeric tables with categorical association scores
and selected outlet/source/payment comparisons, including sample sizes. Slide 6
links each finding to a specific modelling check and gives the majority-class
baseline. This is a later explanation of the evidence, not a revised account
of what was presented.

Both decks keep six slides. Speaker notes contain the prepared spoken story and
handovers, without source lists. Sources stay in the separate handouts.

## Supporting figures retained for review

These figures are unchanged copies from the pinned analysis commits. They give
context for the native charts in the decks; they are not screenshots replacing
editable evidence.

| Figure | Purpose | Source |
| --- | --- | --- |
| [Categorical associations](evidence/categorical_associations.png) | Full ranking behind the expanded deck's selected fields | PR #8, 2c5b3e9 |
| [Quantity and quality](evidence/quantity_quality.png) | Full category comparisons behind the water-supply finding | PR #8, 2c5b3e9 |
| [Management and payment](evidence/management_payment.png) | Context for payment differences and the weaker management score | PR #8, 2c5b3e9 |
| [Numeric correlations](evidence/numeric_spearman_heatmaps.png) | Pairwise counts and zero-height sensitivity | PR #13, 747b6af |
| [Regional missingness](evidence/missingness_by_region.png) | Coverage behind the age limitation | PR #13, 747b6af |
| [Age within regions](evidence/age_by_region.png) | Checks the pooled age pattern within locations | PR #13, 747b6af |
| [Age within extraction classes](evidence/age_by_extraction_type_class.png) | Checks the pooled age pattern within technologies | PR #13, 747b6af |
| [Regional functionality](../../06_region_functionality.png) | Geographic context for H2 | Already in main |
| [Sampled interactive map](../../06_pump_status_map.html) | Shows sampled locations and states | Already in main; 3,000 sampled points |

The geographic plots were reviewed for integration but were not selected as
separate slides in the presented six-slide deck. Geographic variation remains
in the written summary so the team's spatial analysis is not lost.

The copied figures preserve their original legends. The deck uses its own
consistent palette for recreated native evidence. Numeric heatmap colours
represent correlations, not pump states.

## Timing and handovers

| Slide | Speaker | Presented deck seconds | Presented script words | Expanded deck seconds | Expanded script words |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Goodfriend | 25 | 45 | 20 | 41 |
| 2 | Tayaphon | 45 | 86 | 40 | 78 |
| 3 | Morisade | 50 | 92 | 45 | 93 |
| 4 | Tari | 65 | 123 | 65 | 122 |
| 5 | Tayaphon | 45 | 97 | 60 | 120 |
| 6 | Goodfriend | 40 | 85 | 40 | 85 |
| Total | | 270 | 528 | 270 | 539 |

The scripts include transitions and speaker handovers. The target leaves
30 seconds within a five-minute slot. Actual delivery depends on pauses,
speaking speed and handovers.

## Editability and provenance

The presented file contains five native charts, two native correlation tables
and five embedded chart workbooks. The expanded file contains six native charts,
one native table and six chart workbooks. Text is editable. The cover illustration
is the single raster artwork and is clearly conceptual rather than a photograph.

The archived decks are byte-for-byte copies of the reviewed deliverables, not
re-authored exports. Their hashes are in the manifest. Both versions were
opened and rendered in PowerPoint during preparation; representative native
objects were edited in disposable copies. Sources list methods, counts and
input hashes.

The integration branch archives aggregate results and presentation material.
It does not add raw CSVs, a cleaned dataset, local metadata or copies of the
separate analysis implementations.
