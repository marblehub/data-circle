# Sprint 1 completion and review record

Prepared 15 October 2026 for [Issue #3](https://github.com/marblehub/data-circle/issues/3)
and [PR #11](https://github.com/marblehub/data-circle/pull/11).

The project owner confirmed that Sprint 1 was presented and identified the
earlier six-slide deck as the presented version. The archive preserves that file
and labels the expanded deck as a later revision.

This record separates completed technical work from independent peer approval.
It does not claim that the requested reviewers have approved PR #11 or that an
actual presentation duration was measured.

## Acceptance criteria

| Issue #3 criterion | Evidence | Status |
| --- | --- | --- |
| Hypotheses clearly documented | [Framework](sprint1_research_framework.md), H1-H6 with fields, scope and descriptive outcomes in the [summary](sprint1_integration.md#hypothesis-results) | Documented; formal team confirmation can occur in PR review. |
| Research questions clearly defined | Main and supporting questions in the framework; prediction separated from description | Complete. |
| Important analytical decisions recorded | [Decision log](sprint1_decisions.md), including implemented rules and deferred choices | Complete. |
| Team findings integrated into a coherent summary | Quality, categorical, geographic, age and numeric findings in the integration summary | Complete, with pinned references for unmerged source work. |
| Presentation figures clear and relevant | [Archive and selection rationale](presentations/sprint1/README.md), editable decks and supporting source figures | Technical content and visual review complete. |
| Presentation can be delivered within five minutes | Six slides, 528 prepared words, 270-second plan; owner confirms delivery | Timing design supported. Actual elapsed time not independently recorded. |
| All Sprint 1 deliverables reviewed | Technical review inventory below | Technical review complete; independent peer approval remains the normal PR step. |

## Issue task coverage

- [x] Review and refine the research questions.
- [x] Consolidate the hypotheses and map them to available fields.
- [x] Record analytical decisions.
- [x] Review quality, categorical and geospatial findings and the additional checks.
- [x] Identify key findings and state their limits.
- [x] Distinguish association, prediction and causation.
- [x] Integrate selected figures and the written results.
- [x] Archive the prepared presentation, notes, sources and timing.
- [x] Record the owner's confirmation that the presentation was delivered.
- [x] Check the technical deliverable inventory and record remaining questions.
- [ ] Obtain independent peer confirmation of the consolidated hypotheses and PR.

The final unchecked item is not a missing document or analysis. It must be
completed by a human reviewer and must not be recorded as approved on their behalf.

## Technical review inventory

| Deliverable | Scope reviewed | Result / limit |
| --- | --- | --- |
| Quality audit, dictionary and cleaner in main | Dataset dimensions, IDs, missing values, duplicates, suspicious values and cleaning policy | Used in the integrated summary; no raw records copied into these documents. |
| Categorical notebook, helper and figures at 2c5b3e9 | Category denominators, 17 association scores, name grouping and causal wording | Aggregate calculations checked against the local inputs. Weak scores do not justify automatic deletion of fields. |
| Geospatial notebook and outputs in main | Coordinate screens, regional/basin rates, sampled map and age bins | Screen difference and calculated-versus-usable age counts are documented. No claim that the map includes every point. |
| Additional analysis and export at 747b6af | Missingness, pairwise numeric checks, age exclusions, strata, standardisation and export policy | Summary numbers independently recalculated. Source implementation remains in PR #13. |
| Presented deck | Six slides, notes, numeric claims, state colours, source handout and editability | Earlier reviewed version archived unchanged; owner identifies it as presented. |
| Expanded revision | Categorical additions, sample sizes, possible leakage and Sprint 2 plan | Later reviewed version archived unchanged and labelled separately. |
| Integration package | Questions, hypotheses, outcomes, decisions, links, provenance and handover | Completes the documentation scope; no trained-model performance is claimed. |

Review references use immutable source commits. Technical review of findings is
not approval or merging of the separate source PRs.

## Validation record

Validation on 15 October 2026:

- Independently recalculated aggregate counts and rates from the original local
  inputs: 59,400 records, target counts, age coverage and bins, missing-year
  denominators, 17 categorical scores, selected category and regional rates,
  pairwise Spearman sensitivity and the two standardised age gaps.
- Confirmed every eligible extreme-age contrast: 16 regions and five extraction
  classes, with at least 30 records in each extreme group.
- Confirmed the two coordinate screens return 1,812 and 1,819 failures.
- Matched the original input hashes to the pinned additional EDA results.
- Verified all archived asset checksums; copied figures match their source
  Git blobs exactly. Both decks match the previously reviewed deliverables.
- Both PowerPoint packages pass ZIP integrity checks. Each has six slides.
  The presented deck retains five native charts, two tables and five chart
  workbooks; the expanded deck retains six charts, one table and six workbooks.
- Verified the prepared notes contain no URL or source-list sections. Preserved
  528 and 539 spoken-word counts and the 270-second plans.
- Checked local Markdown links in the new integration documents, archive and
  README: 56 valid targets before this final record was inserted.
- Existing integration-branch regression suite: 18 tests passed.
- Ruff from the WSL team project root: all checks passed.
- Reviewed the allowed staging list and patch whitespace. No raw or cleaned
  dataset, local export metadata, temporary build files or private task folder
  is part of the integration package.

Commands for the existing code checks, from the team project root:

```bash
uv run pytest -q
uv run ruff check src tests
git diff --cached --check
```

The existing project virtual environment was used for these checks. No runtime
code, dependency specification or lockfile changed. GitHub CI and human review
are separate from this local validation record.

The [evidence manifest](presentations/sprint1/evidence_manifest.json) contains
deck/figure hashes and original source paths. The [aggregate evidence](sprint1_evidence.json)
contains counts, denominators and source-input hashes without waterpoint-level data.

## Source PRs and merge boundary

- Data quality Issue #4 and geospatial Issue #6 are already integrated into main.
- Categorical EDA [PR #8](https://github.com/marblehub/data-circle/pull/8) is separately open at 2c5b3e9.
- Additional EDA and export [PR #13](https://github.com/marblehub/data-circle/pull/13) are separately open at 747b6af.
- PR #11 integrates their documented findings and immutable aggregate snapshots;
  it does not merge their code or close their issues.

Reviewing or merging those source PRs remains separate work. If they change,
compare new results with the pinned evidence before replacing the archived record.

## Handover and remaining human steps

The owner will change PR #11 from draft to ready for review. The three existing
requested reviewers can then confirm the framing, source interpretations and
deliverable record through the normal review process. No approval is invented
from the fact that the team presented.

After review and any requested corrections, merge according to the team workflow.
Keep `Closes #3` because this package now supplies the research integration and
presentation record; it should close only when the PR is merged.

The Sprint 2 questions in the decision log remain open by design: feature
availability, uncertain source meanings, aliases, validation splits and model
comparisons are not claimed as completed Sprint 1 work.
