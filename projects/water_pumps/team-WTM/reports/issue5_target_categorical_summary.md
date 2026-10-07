# Issue #5 – Target-class and categorical EDA: summary (PM-3)

Notebook: `notebooks/02_target_categorical_eda_taya.ipynb` · Figures: `reports/figures/5_*.png`

All results are **associations**, not proof of cause.

## Target class

| Class | Count | % |
|---|---|---|
| functional | 32,259 | 54.3 |
| functional needs repair | 4,317 | 7.3 |
| non functional | 22,824 | 38.4 |

Clear imbalance: the majority class is ~7.5× larger than "needs repair". Accuracy alone is misleading (always predicting "functional" gives ~54%).

## Strongest categorical associations (Cramér's V)

quantity 0.31 · waterpoint_type 0.25 · extraction_type 0.25 · funder 0.20 · payment_type 0.18 · installer 0.18 · source 0.15 · water_quality 0.14 · management 0.13 · permit 0.03

## Key patterns

- **Quantity:** 97% of *dry* points are non functional (possible leakage – direction unclear).
- **Extraction / water-point type:** "other" ≈ 80% non functional; gravity, handpump, rope pump ≈ 60–65% functional; "communal standpipe multiple" 53% non functional vs 30% for single.
- **Source:** lakes 77% and dams 58% non functional; springs and rainwater harvesting best. Surface sources have ~2× the needs-repair share of groundwater.
- **Payment:** never pay 48% non functional vs annually 18% (direction unclear).
- **Management:** VWC (68% of points) 43% non functional vs water board / WUA / private operator ~17–23%.
- **Installer / funder:** 2,000+ messy names; large installers range ~25% to ~70% non functional.
- **Needs repair** is highest (12–21%) for salty-abandoned water, surface sources, water authority / WUG / parastatal management.

## Hypotheses

| Hypothesis | Result |
|---|---|
| Pump / extraction type associated with status | ✅ supported |
| Water quantity / quality associated with status | ✅ strongly supported |
| Management associated with status | ⚠️ partly (payment clear, management type weaker) |
| Installer / funder informative | ✅ supported after cleaning |
| Water source associated with status | ✅ moderate |
| Permit associated with status | ❌ challenged |

## For Sprint 2 (modelling)

1. Per-class precision/recall/F1, stratified CV, class weights for "needs repair".
2. Test models with and without `quantity` and "unknown" categories (leakage risk).
3. Keep one column per redundant group; drop `quantity_group`, `recorded_by`.
4. Clean + group `installer`, `funder`, `lga` (top-N + "other"); drop `wpt_name`, `subvillage`, `scheme_name`, `ward`.

---

## Talk script for my part (~1.5 min)

1. **`5_01_target_distribution.png`** – "54% of water points work, 38% do not, and only 7% need repair. This imbalance means we must look at per-class results, not only accuracy."
2. **`5_02_cramers_v_ranking.png`** – "Of the categorical features, water quantity, water-point type and extraction type are most strongly associated with status. Permit shows almost no association."
3. **`5_05_quantity_quality.png`** – "97% of dry points are non functional. But a pump may be called dry because it is broken, so this may be leakage."
4. **`5_06_management_payment.png`** – "Where people pay for water, more points work. We cannot say which way this goes – payment may fund repairs, or people may stop paying after a breakdown."

Closing: "These signals and the cleaning needs go into the Sprint 2 pipeline."
