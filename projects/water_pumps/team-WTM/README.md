# 💧 Team WTM - Water Pump Functionality Prediction

**ReDI School Data Circle 2026**

> A team data-science project investigating Tanzanian water-pump functionality using exploratory data analysis, geospatial analysis, and machine learning, with a focus on producing useful insights for maintenance prioritisation.

**Current status:** 🟡 Sprint 1: Exploratory Data Analysis

---

## Project Overview

Reliable access to clean water depends not only on installing water infrastructure, but also on identifying pumps that are failing, require repair, or are no longer operational.

This project uses data from the **DrivenData "Pump it Up: Data Mining the Water Table"** challenge to investigate water-pump functionality across Tanzania.

The dataset originates from **Taarifa and the Tanzanian Ministry of Water**.

Our goal is not only to build an accurate classification model, but also to understand the factors associated with pump functionality and explore how data-driven insights may support maintenance planning and prioritisation.

---

## Problem Statement

Each water point belongs to one of three operational classes:

| Status | Meaning |
|---|---|
| `functional` | The pump is operational |
| `functional needs repair` | The pump operates but requires repair |
| `non functional` | The pump is not operational |

This is therefore a **supervised multiclass classification problem.**

The prediction workflow can be summarized as:

Input features (X) -> Machine-learning model -> Predicted pump status (status_group)

The input features include geographical, technical, management, water-resource, and temporal information about each water point.

---

## Project Objectives

The project has four main objectives:

1. Explore and assess the quality of the Tanzanian water-point dataset.
2. Identify factors associated with water-pump functionality.
3. Develop and compare machine-learning classification models.
4. Communicate useful findings that may support maintenance prioritisation.

Where feasible, the final results will also be presented through an interactive dashboard.

---

## Research Questions

Initial research questions include:

- Are older pumps more likely to require repair or become non-functional?
- Does pump or extraction type influence functionality?
- Are particular regions or basins associated with higher failure rates?
- How are water quantity and water quality related to pump status?
- Does management structure show an association with pump functionality?
- Does installer or funder information provide useful predictive information?
- Which features are most useful for distinguishing pumps that need repair from fully functional pumps?
- Can geographical and predictive information help identify areas where maintenance attention may be most valuable?

These questions may be refined as the exploratory analysis progresses.

---

## Data

The dataset contains more than 40 variables describing water points across Tanzania.

Features include:

- geographical coordinates;
- region and basin;
- construction year;
- installer and funder;
- water source;
- water quantity and quality;
- extraction method;
- pump type;
- management structure;
- payment information;
- population served;
- recording date.

### Data Source

The project is based on the DrivenData competition:

**Pump it Up: Data Mining the Water Table**

https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/

The original dataset is **not stored in this repository**.

You can obtain the data from the official source and place the files locally under:

```text
data/raw/
```

Processed datasets should be stored locally under:

```text
data/processed/
```

CSV, ZIP, Parquet, Excel, pickle, and similar data files are excluded from Git through `.gitignore`.

The repository's MIT licence applies to the project code only and does not relicense the original dataset.

---

# Team

| Role | Primary Responsibility |
|---|---|
| **PM-1** | Project coordination, integration, GitHub/project management, shared technical work |
| **PM-2** | Data quality, preprocessing, and supporting analysis |
| **PM-3** | Exploratory analysis, visualisation, and geospatial analysis |
| **PM-4** | Modelling, evaluation, model interpretation, and deployment support |

These responsibilities are not strictly fixed. All team members are expected to:

- understand the overall project;
- participate in code reviews;
- contribute to discussions and sprint reviews;
- document their work;
- support other team members where necessary;
- gain exposure to different parts of the data-science workflow.

---

# Project Plan

The project follows three main sprints.

## Sprint 1: Exploratory Data Analysis

**Weeks 1-3**

Main activities:

- development-environment setup;
- dataset documentation;
- data-quality assessment;
- missing-value analysis;
- duplicate and invalid-value investigation;
- exploratory data analysis;
- target-class analysis;
- categorical-variable investigation;
- geospatial analysis;
- hypothesis definition;
- research-question refinement;
- initial feature engineering.

### Sprint 1: Deliverables

- hypotheses and research questions;
- data-quality findings;
- exploratory visualisations;
- documented analytical decisions;
- reproducible notebooks and scripts;
- 5-minute sprint presentation.

---

## Sprint 2: Model Development

**Weeks 4-6**

Main activities:

- preprocessing pipeline development;
- feature engineering;
- feature selection where appropriate;
- baseline modelling;
- model comparison;
- cross-validation;
- hyperparameter tuning where justified;
- model evaluation.

Potential models include:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- other justified classification methods.

### Evaluation Metrics

Model performance will be assessed using:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- class-specific performance.

Particular attention will be given to the `functional needs repair` class because identifying these pumps may be relevant for preventive maintenance.

### Sprint 2: Deliverables

- processed modelling dataset;
- reproducible preprocessing and modelling pipeline;
- model comparison;
- documented evaluation results;
- interpretation of model performance;
- 5-minute sprint presentation.

---

## Sprint 3: Insights and Deployment

**Weeks 7-9**

Main activities:

- model interpretation;
- feature-importance analysis;
- SHAP or related explainability methods where appropriate;
- investigation of nonlinear relationships and interactions;
- comparison of findings with initial hypotheses;
- geospatial presentation of results;
- analysis of model and data limitations;
- maintenance-oriented interpretation;
- optional Streamlit dashboard;
- final presentation.

### Sprint 3: Deliverables

- interpretation of model results;
- key findings and limitations;
- complete GitHub repository;
- optional interactive dashboard;
- final presentation.

---

# Repository Structure

```text
team-WTM/
├── app/
│   └── ...
│
├── data/
│   ├── raw/
│   │   └── data_dictionary.md
│   ├── processed/
│   │   └── processed_data_dictionary.md
│
├── notebooks/
│   └── ...
│
├── reports/
│   ├── figures/
│   └── ...
│
├── src/
│   └── team_wtm/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── modeling.py
│       └── evaluation.py
│
├── tests/
│   └── ...
│
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── workflow.md
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

### `data/`

Contains local raw and processed datasets together with version-controlled documentation describing the data.

Actual datasets are not committed to GitHub.

### `notebooks/`

Contains exploratory Jupyter notebooks.

Notebook names should be descriptive and sequential, for example:

```text
01_target_distribution_pm2.ipynb
02_data_quality_pm2.ipynb
03_pump_age_analysis_pm3.ipynb
04_geospatial_analysis_pm3.ipynb
```

Exploratory notebooks should not become the permanent home of reusable project logic.

### `src/team_wtm/`

Contains reusable Python code for:

- loading data;
- cleaning;
- feature engineering;
- preprocessing;
- modelling;
- evaluation.

### `reports/`

Contains:

- figures;
- sprint materials;
- presentation resources;
- final project outputs.

### `app/`

Reserved for the optional Streamlit application.

### `tests/`

Contains tests for reusable project code.

---

# Environment Setup

The project uses **Python 3.12** and **uv** for Python environment and dependency management.

## Clone the Team Fork

```bash
git clone https://github.com/marblehub/data-circle.git
cd data-circle
```

For SSH:

```bash
git clone git@github.com:marblehub/data-circle.git
cd data-circle
```

Add the original ReDI repository as the upstream remote:

```bash
git remote add upstream https://github.com/ReDI-School/data-circle.git
```

Check the configuration:

```bash
git remote -v
```

Expected structure:

```text
origin    -> marblehub/data-circle
upstream  -> ReDI-School/data-circle
```

Enter the Team WTM project directory:

```bash
cd projects/water_pumps/team-WTM
```

---

## Install Dependencies

Install the project environment:

```bash
uv sync
```

Check the Python version:

```bash
uv run python --version
```

The project currently targets:

```text
Python 3.12
```

Run Python commands through the project environment using:

```bash
uv run python
```

Run Jupyter using:

```bash
uv run jupyter lab
```

---

# Git and GitHub Workflow

The team uses a **feature-branch workflow**.

The `main` branch represents the stable integrated version of the project.

Team members should normally **not develop directly on `main`**.

---

## 1. Start From an Updated `main`

Before starting new work:

```bash
git switch main
git pull origin main
```

---

## 2. Work From a GitHub Issue

Each meaningful task should first exist as a GitHub Issue.

Example:

```text
Issue #5
Analyse pump age versus functionality
```

---

## 3. Create a Branch for the Issue

Use:

```text
feature/<issue-number>-<description>
```

Example:

```bash
git switch -c feature/5-pump-age-analysis
```

Other examples:

```text
feature/7-geospatial-analysis
feature/10-data-cleaning
docs/12-update-data-dictionary
fix/15-construction-year-handling
```

Note:
Branches should describe the work, not the person performing it.

---

## 4. Make Small Commits

Example:

```bash
git add .
git commit -m "feat: analyse pump age and functionality"
```

Recommended prefixes include:

```text
feat:     new functionality or analysis
fix:      bug or data-handling correction
docs:     documentation
test:     tests
refactor: code restructuring
chore:    project maintenance
```

---

## 5. Push the Branch

```bash
git push -u origin feature/5-pump-age-analysis
```

---

## 6. Create a Pull Request

The Pull Request should target:

```text
marblehub/data-circle:main
```

not the original:

```text
ReDI-School/data-circle:main
```

unless the ReDI instructors explicitly request a submission to their repository.

Each Pull Request should:

- explain what changed;
- reference the corresponding Issue;
- describe how the work was validated;
- be reviewed by at least one other team member.

Example:

```text
Closes #5
```

can be added to the PR description so the Issue closes automatically after merging.

---

## 7. Peer Review

Every meaningful Pull Request should be reviewed by another team member.

Suggested review rotation:

```text
PM-1 -> PM-2
PM-2 -> PM-3
PM-3 -> PM-4
PM-4 -> PM-1 or PM-2
```

Reviewers should check:

- correctness;
- clarity;
- reproducibility;
- documentation;
- unnecessary duplication;
- whether acceptance criteria are satisfied.

---

## 8. Merge

Prefer:

```text
Squash and merge
```

for feature branches.

After merging:

```bash
git switch main
git pull origin main
git branch -d feature/5-pump-age-analysis
git fetch --prune
```

---

# Upstream Synchronisation

The original ReDI School repository is configured as:

```text
upstream
```

while the Team WTM fork is:

```text
origin
```

PM-1 will normally coordinate synchronisation with the ReDI upstream repository.

Example:

```bash
git switch main
git fetch upstream
git merge upstream/main
git push origin main
```

Other team members can then update normally using:

```bash
git switch main
git pull origin main
```

---

# Project Management

The project uses GitHub Issues, milestones, Pull Requests, and a simple project board.

## Milestones

Work is grouped into:

- **Sprint 1 — EDA**
- **Sprint 2 — Modelling**
- **Sprint 3 — Insights & Deployment**

## Suggested Issue Labels

```text
data
eda
geospatial
modeling
dashboard
documentation
bug
blocked
```

## Project Board

Suggested workflow:

```text
Backlog
   ↓
Ready
   ↓
In Progress
   ↓
Review
   ↓
Done
```

Each task (should normally) have:

- one primary owner;
- one sprint milestone;
- an appropriate label;
- clear acceptance criteria.

---

# Development Principles

Team WTM aims to:

- keep all analysis reproducible;
- protect the stability of `main`;
- make small and reviewable changes;
- use Issues to define work;
- use Pull Requests for integration;
- review one another's code and analysis;
- document important analytical decisions;
- avoid committing raw datasets or secrets;
- separate exploratory notebooks from reusable Python code;
- distinguish correlation from causation;
- evaluate models using more than accuracy alone;
- focus on actionable and defensible insights;
- keep the project understandable to someone outside the team.

---

# Project Direction

The project is framed around:

> **Using data and machine learning to understand Tanzanian water-pump functionality and explore how predictive and geographical information may support maintenance prioritisation.**

The intended progression is:

```text
Understand the data
        ↓
Identify functionality patterns
        ↓
Build predictive models
        ↓
Interpret predictions
        ↓
Identify maintenance-relevant insights
```

---

# Limitations

Model results will be interpreted carefully.

A feature being predictive does not necessarily mean it causes pump failure.

For example, if installer information is associated with pump status, this alone does not establish that an installer caused the observed failures. Differences may also reflect geography, pump age, technology, environmental conditions, or other unobserved factors.

The project therefore distinguishes between:

- prediction;
- statistical association;
- causal explanation.

---

# License

Project code is released under the [MIT License](LICENSE).

The underlying Tanzanian water-point dataset remains subject to the license and terms of its original providers.

---

# Acknowledgements

The project is based on the DrivenData **Pump it Up: Data Mining the Water Table** challenge.

Data originate from **Taarifa and the Tanzanian Ministry of Water**.

This project is being developed as part of the **ReDI School Data Circle 2026**.

Original ReDI School repository:

https://github.com/ReDI-School/data-circle