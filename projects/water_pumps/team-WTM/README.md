# Water Pump Functionality Prediction

## ReDI School Data Circle 2026

This project investigates the functionality of water pumps across Tanzania using exploratory data analysis, geospatial analysis, and machine learning.

The project is based on the **DrivenData "Pump it Up: Data Mining the Water Table"** challenge and uses water-point data from **Taarifa and the Tanzanian Ministry of Water**.

The work is being completed as part of the **ReDI School Data Circle 2026**.

## Problem Statement

Reliable access to clean water depends not only on installing water infrastructure, but also on identifying pumps that are failing or require maintenance.

The goal of this project is to use available information about water points to classify pumps into three operational states:

- `functional`
- `functional needs repair`
- `non functional`

Beyond prediction accuracy, the project aims to understand the factors associated with pump functionality and explore how data-driven insights could support maintenance prioritisation.

## Project Objectives

The project has four main objectives:

1. Explore and clean the Tanzanian water-point dataset.
2. Identify factors associated with water pump functionality.
3. Develop and compare machine-learning classification models.
4. Communicate actionable insights through visualisations and, if feasible, an interactive dashboard.

## Research Questions

Initial questions include:

- Are older pumps more likely to be non-functional or require repair?
- Does pump type influence functionality?
- Are some geographical regions associated with higher failure rates?
- How are water quantity and water quality related to pump functionality?
- Does installer or management structure show an association with pump condition?
- Which features are most useful for distinguishing pumps that need repair from fully functional pumps?
- Can model predictions help identify pumps or geographical areas that may deserve maintenance attention?

These questions may be refined as the exploratory analysis progresses.

## Target Variable

The target variable is `status_group`, with three classes:

| Class | Description |
|---|---|
| `functional` | Pump is operational |
| `functional needs repair` | Pump operates but requires repair |
| `non functional` | Pump is not operational |

This is therefore a **multiclass supervised classification problem**.

## Data

The dataset contains more than 40 variables describing water points, including:

- geographical location;
- construction year;
- water source;
- water quality and quantity;
- extraction method;
- pump type;
- installer and funder;
- management structure;
- payment information;
- population served.

### Data source

The project uses data from the DrivenData competition:

**Pump it Up: Data Mining the Water Table**

https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

The original data files are **not stored in this repository**.

Kindly obtain the dataset from the official source and place it locally in the `data/` directory.

The dataset remains subject to the terms and conditions of its original providers. The repository's MIT licence applies to the project code and does not relicense the original dataset.

## Project Plan

The project follows three main sprints.

### Sprint 1: Exploratory Data Analysis

**Weeks 1–3**

Main activities:

- project and environment setup;
- data quality assessment;
- exploratory data analysis;
- missing-value investigation;
- categorical-variable analysis;
- geospatial exploration;
- hypothesis definition;
- research-question refinement;
- initial feature engineering.

Deliverable:

**EDA findings, reproducible analysis, visualisations, and a 5-minute sprint presentation.**

### Sprint 2: Model Development

**Weeks 4–6**

Main activities:

- preprocessing pipeline development;
- feature engineering;
- baseline modelling;
- model comparison;
- cross-validation;
- hyperparameter tuning where appropriate;
- model evaluation.

Potential models include:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- other justified classification methods.

Evaluation will consider:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- class-specific performance.

Deliverable:

**Reproducible modelling pipeline, model comparison, interpretation, and a 5-minute sprint presentation.**

### Sprint 3: Insights and Communication

**Weeks 7–9**

Main activities:

- model interpretation;
- feature importance;
- SHAP or related explainability methods where appropriate;
- comparison with initial hypotheses;
- geospatial communication of results;
- analysis of model limitations;
- dashboard development if feasible;
- final presentation.

The final objective is not only to produce predictions, but also to communicate useful and defensible insights from the data.

## Repository Structure

```text
.
├── data/
│   ├── data_dictionary.md
│   └── processed_data_dictionary.md
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── app/
├── tests/
├── README.md
├── workflow.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .gitignore
└── LICENSE
```

### `data/`

Contains documentation describing the raw and processed data.

Raw datasets are kept locally and are not committed to GitHub.

### `notebooks/`

Contains exploratory Jupyter notebooks.

Notebooks should be numbered sequentially and named clearly, for example:

```text
01_exploratory_data_analysis_name.ipynb
02_data_cleaning_name.ipynb
03_feature_engineering_name.ipynb
```

### `src/`

Contains reusable Python code for data processing, feature engineering, modelling, and other project functions.

### `reports/`

Contains figures, sprint material, and final presentation resources.

### `app/`

Reserved for an optional Streamlit application.

## Environment Setup

This project uses Python and supports `uv` for environment and dependency management.

Clone the repository:

```bash
git clone https://github.com/marblehub/water_pump_p26.git
cd water_pump_p26
```

Install the project dependencies:

```bash
uv sync
```

Run Jupyter:

```bash
uv run jupyter lab
```

A `requirements.txt` file is also maintained for compatibility with the ReDI project requirements and other Python environments.

## Git Collaboration Workflow

The project uses a **feature-branch workflow**.

Do not normally develop directly on `main`.

For each GitHub issue:

```bash
git switch main
git pull
git switch -c feature/<issue-number>-<short-description>
```

Example:

```bash
git switch -c feature/12-pump-age-analysis
```

Make small, focused commits:

```bash
git add .
git commit -m "feat: add pump age analysis"
```

Push the branch:

```bash
git push -u origin feature/12-pump-age-analysis
```

Then open a Pull Request to `main`.

Every Pull Request should be reviewed by at least one other team member before merging.

## Team

| Team member | GitHub | Responsibilities |
|---|---|---|
| Project Manager | `Goodfriend Whyte <@marblehub>` | Coordination and shared technical work |
| Team member | TBD | TBD |
| Team member | TBD | TBD |
| Team member | TBD | TBD |

## Working Principles

We aim to:

- keep analysis reproducible;
- make small and reviewable changes;
- document important analytical decisions;
- avoid committing raw datasets or secrets;
- separate exploratory notebooks from reusable code;
- review one another's work;
- distinguish statistical associations from causal conclusions;
- keep the project focused on useful insights rather than model accuracy alone.

## License

The project code is released under the MIT License in [LICENSE.md](LICENSE.md).

The underlying water-point dataset remains subject to the license and terms of its original providers.

## Acknowledgements

Data are provided through the DrivenData **Pump it Up: Data Mining the Water Table** challenge and originate from Taarifa and the Tanzanian Ministry of Water.

Project developed as part of the **ReDI School Data Circle 2026**.
