## 1. Identify the Problem

Access to clean and reliable water depends not only on installing waterpoints but also on keeping them operational. In Tanzania, some recorded waterpoints are fully functional, some are still operating but require repair, and others are completely non-functional.

The practical problem is therefore to identify characteristics associated with waterpoint condition and determine whether those characteristics can be used to predict which waterpoints are likely to be faulty.

DrivenData frames the task as predicting the operating condition of each waterpoint using information about the type of waterpoint, when it was installed, how it is managed, and other geographical, technical, and operational characteristics. Better identification of faulty waterpoints could support maintenance planning and help improve access to clean, potable water.

The prediction target, `status_group`, contains three classes:

- `functional` — operational and does not require repair
- `functional needs repair` — operational but requires repair
- `non functional` — not operational

The central project problem can therefore be expressed as:

> **Can the available characteristics of Tanzanian waterpoints be used to understand differences in their operating condition and accurately predict whether a waterpoint is functional, needs repair, or is non-functional?**

---

## 2. Review Existing Knowledge

The dataset comes from the Taarifa waterpoints dashboard, which aggregates information from the Tanzanian Ministry of Water. Taarifa was designed as a platform for reporting and managing infrastructure-related issues.

The official dataset already provides several groups of information that may help explain or predict waterpoint condition.

**Geographical information** includes:

- `longitude`
- `latitude`
- `gps_height`
- `basin`
- `region`
- `region_code`
- `district_code`
- `lga`
- `ward`
- `subvillage`

**Construction and organisational information** includes:

- `construction_year`
- `funder`
- `installer`
- `scheme_management`
- `scheme_name`
- `management`
- `management_group`
- `permit`
- `public_meeting`

**Water and infrastructure characteristics** include:

- `extraction_type`
- `extraction_type_group`
- `extraction_type_class`
- `water_quality`
- `quality_group`
- `quantity`
- `quantity_group`
- `source`
- `source_type`
- `source_class`
- `waterpoint_type`
- `waterpoint_type_group`

Other variables include `population`, `payment`, `payment_type`, `date_recorded`, `amount_tsh`, `wpt_name`, and `recorded_by`.

What is **already known** is therefore the type of information available and the prediction target.

What is **not yet known from the dataset description** is:

- which characteristics are most strongly associated with waterpoint condition;
- whether failures are geographically concentrated;
- whether older waterpoints have poorer operating conditions;
- whether particular extraction technologies or waterpoint types perform differently;
- whether water availability, quality, management, payment arrangements, or population are associated with functionality;
- which variables contain genuinely useful predictive information;
- how well the three classes, particularly `functional needs repair`, can be distinguished.

These unanswered questions motivate the exploratory analysis and later modelling.

---

## 3. Define Research Questions and Hypotheses

### Main research question

> **Which geographical, technical, environmental, and management characteristics are associated with the operating condition of Tanzanian waterpoints, and how effectively can these characteristics predict waterpoint status?**

### Supporting research questions

1. What is the distribution of the three waterpoint-status classes?
2. How does waterpoint functionality vary geographically across regions, basins, districts, and other locations?
3. Is waterpoint age associated with operating condition?
4. Does water availability, represented by variables such as `quantity` and `quantity_group`, differ between functional and non-functional waterpoints?
5. Are some extraction technologies or waterpoint types associated with better or worse operating conditions?
6. Do management arrangements, payment systems, installers, or funders show differences in waterpoint functionality?
7. Which variables provide the strongest predictive information after accounting for related or redundant features?
8. How accurately can machine-learning models distinguish among the three operating conditions?
9. How well can the models identify the particularly important `functional needs repair` class?

### Initial hypotheses

**H1:** Older waterpoints are more likely to be non-functional or require repair than newer waterpoints.

**H2:** Waterpoint functionality varies geographically across Tanzania.

**H3:** Waterpoints recorded as having enough water are more likely to be functional than those recorded as insufficient or dry.

**H4:** Waterpoint condition differs among extraction technologies and waterpoint types.

**H5:** Management and payment arrangements are associated with differences in waterpoint functionality.

**H6:** A combination of geographical, technical, water-related, and management variables will provide better prediction of waterpoint status than any single characteristic.

These hypotheses describe relationships to investigate. Because the dataset is observational rather than a controlled experiment, an observed relationship should not automatically be interpreted as causal.

---

## 4. Choose a Methodology

The project uses an **observational, exploratory, and predictive data-science methodology**.

It is not an experimental study because the team does not manipulate waterpoint conditions. Instead, existing waterpoint records are analysed to identify patterns and build predictive models.

The methodology consists of four main stages.

**Exploratory Data Analysis**

EDA will be used to understand:

- target-class distribution;
- missing and unusual values;
- distributions of numerical and categorical variables;
- geographical patterns;
- relationships between individual features and `status_group`.

**Feature preparation and engineering**

Some variables represent different levels of the same underlying concept. For example, the official data contain:

```text
extraction_type
extraction_type_group
extraction_type_class
```

and similarly:

```text
source
source_type
source_class
```

as well as pairs such as:

```text
quantity / quantity_group
water_quality / quality_group
payment / payment_type
waterpoint_type / waterpoint_type_group
management / management_group
```

These relationships should be examined carefully to avoid unnecessary redundancy in later models.

Derived variables can also be created where meaningful. One important example is:

```text
pump_age = year(date_recorded) - construction_year
```

provided invalid or unknown construction years are handled appropriately.

**Predictive modelling**

The task is a supervised **multiclass classification** problem.

Candidate models include:

- Logistic Regression as a baseline;
- Random Forest;
- Gradient Boosting;
- additional classifiers if they provide a meaningful comparison.

Preprocessing and modelling should be implemented through reproducible scikit-learn pipelines where possible.

**Evaluation**

Models should be evaluated using:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- ROC-AUC where appropriate.

Class-specific metrics are important because overall accuracy alone may hide poor performance for a less common class such as `functional needs repair`.

---

## 5. Collect and Prepare the Data

The project uses the data supplied for DrivenData's **Pump it Up: Data Mining the Water Table** challenge. The features describe Tanzanian waterpoints, while the target identifies their operating condition. The competition data originate from the Taarifa waterpoints dashboard and Tanzania Ministry of Water information.

The preparation stage should first establish the basic structure of the dataset:

- number of observations and features;
- target distribution;
- numerical, categorical, boolean, date, and identifier variables;
- duplicated records;
- missing values;
- invalid or suspicious values;
- categorical cardinality;
- relationships among duplicated or hierarchical feature representations.

Particular attention should be paid to variables that can contain values requiring interpretation, such as:

```text
construction_year
population
longitude
latitude
gps_height
amount_tsh
```

For example, a zero `construction_year` should not automatically be interpreted as a waterpoint constructed in year zero. It should first be investigated as a possible missing or unknown value.

Categorical variables such as `installer`, `funder`, `wpt_name`, `scheme_name`, `subvillage`, and `ward` may also contain many unique values and therefore require careful treatment before modelling.

The preparation process should preserve the original raw data and document all cleaning and transformation decisions so that the analysis remains reproducible.

---

## 6. Analyze the Data

The analysis should move from **general understanding → individual relationships → combined predictive signals**.

### Target analysis

First examine the distribution of:

```text
functional
functional needs repair
non functional
```

using counts and percentages.

This establishes whether the target is imbalanced and provides the baseline against which later model performance should be interpreted.

### Geographical analysis

Use variables such as:

```text
latitude
longitude
region
basin
lga
district_code
ward
```

to investigate whether functionality varies spatially.

Questions include:

- Are non-functional waterpoints concentrated in particular regions?
- Are some basins associated with higher proportions of failed waterpoints?
- Are there areas with unusually high proportions of waterpoints needing repair?

### Infrastructure and technology analysis

Compare `status_group` with:

```text
extraction_type
extraction_type_group
extraction_type_class
waterpoint_type
waterpoint_type_group
construction_year
```

to investigate whether particular technologies, waterpoint designs, or ages show different functionality patterns.

### Water-resource analysis

Investigate:

```text
quantity
quantity_group
water_quality
quality_group
source
source_type
source_class
```

to determine whether water availability, quality, or source characteristics differ among the three operating conditions.

### Management and financial analysis

Examine:

```text
management
management_group
scheme_management
payment
payment_type
permit
public_meeting
```

to determine whether organisational or payment characteristics are associated with functionality.

### Installer and funder analysis

Investigate whether functionality differs among `installer` and `funder` categories while taking their potentially high cardinality and uneven sample sizes into account.

### Temporal and derived-feature analysis

Use `construction_year` together with `date_recorded` to investigate waterpoint age.

Rather than considering construction year alone, a derived age variable may provide a more interpretable measure of how long a waterpoint had existed when its condition was recorded.

### From EDA to predictive modelling

The final purpose of Sprint 1 analysis is not merely to produce plots. It should identify:

- reliable patterns;
- questionable data;
- potentially useful features;
- redundant features;
- features requiring transformation;
- hypotheses worth testing further;
- potential predictive signals.

These findings then provide the evidence for deciding what should enter the modelling pipeline in Sprint 2.

The analytical interpretation should remain careful: **EDA can establish associations and patterns, while the later classification models test predictive usefulness. Neither by itself proves that a particular feature causes a waterpoint to fail.**