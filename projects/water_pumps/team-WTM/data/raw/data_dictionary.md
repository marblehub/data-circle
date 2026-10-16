# Data Description

This water pump dataset is sourced from Tanzania. The goal is to predict the functionality status of each waterpoint.

- `training_set_values` contains information about each waterpoint that can help explain whether the pump is functioning.
- `training_set_labels` contains the target variable to predict.
- Both datasets contain an `id` column, which connects each label to the corresponding waterpoint and allows the datasets to be merged.

The two datasets will be merged so that information about each waterpoint and its corresponding functionality status is contained in one dataset. This makes it easier to perform data quality checks and exploratory analysis.

## The Merged Dataset

The merged dataset contains 41 columns and 59,400 rows. Each row represents a waterpoint. The columns describe its location, construction, management, water source, water quality, population, and functionality status.

The dataset contains both numerical and categorical variables, with 40 input columns (including the `id` identifier) and one target variable, `status_group`.

| Column | Description |
| --- | --- |
| `id` | Identifier connecting each waterpoint to its functionality label. |
| `amount_tsh` | Total static head (amount of water available to the waterpoint). |
| `date_recorded` | Date the row was entered. |
| `funder` | Who funded the well; literal `"0"` values are suspected unknown-value placeholders. |
| `gps_height` | Altitude of the well; frequent zeros may be placeholders, while negative elevations can be valid. |
| `installer` | Organization that installed the well; check placeholder values and inconsistent spellings. |
| `longitude` | GPS longitude coordinate in degrees; zero paired with near-zero latitude is a suspicious location. |
| `latitude` | GPS latitude coordinate in degrees; negative values indicate locations south of the equator. |
| `wpt_name` | Name of the waterpoint, if available. |
| `num_private` | Meaning is not established in this project. Has 65 distinct values, including 58,643 zeros; dropped by the current cleaning rule, with an option to retain it. |
| `basin` | Geographic water basin. |
| `subvillage` | Geographic location. |
| `region` | Geographic location. |
| `region_code` | Geographic location (coded). |
| `district_code` | Geographic location (coded). |
| `lga` | Geographic location. |
| `ward` | Geographic location. |
| `population` | Population around the well; zeros are treated as unknown in the default cleaner. |
| `public_meeting` | True/false indicator. |
| `recorded_by` | Group entering the row of data. |
| `scheme_management` | Who operates the waterpoint. |
| `scheme_name` | Name of the water scheme associated with the waterpoint. |
| `permit` | Whether the waterpoint is permitted. |
| `construction_year` | Year the waterpoint was constructed; zero indicates an unknown year under the project convention. Nonzero years after recording are flagged for review. |
| `extraction_type` | Kind of extraction the waterpoint uses. |
| `extraction_type_group` | Kind of extraction the waterpoint uses (group). |
| `extraction_type_class` | Kind of extraction the waterpoint uses (class). |
| `management` | How the waterpoint is managed. |
| `management_group` | How the waterpoint is managed (group). |
| `payment` | What the water costs. |
| `payment_type` | What the water costs (type). |
| `water_quality` | Quality of the water. |
| `quality_group` | Quality of the water (group). |
| `quantity` | Quantity of water. |
| `quantity_group` | Quantity of water (group). |
| `source` | Source of the water. |
| `source_type` | Source of the water (type). |
| `source_class` | Source of the water (class). |
| `waterpoint_type` | Kind of waterpoint. |
| `waterpoint_type_group` | Kind of waterpoint (group). |
| `status_group` | **Target variable:** functionality status of the water pump. |

## Target Variable

`status_group` has three classes in the training labels:

| Class | Rows | Meaning |
| --- | ---: | --- |
| `functional` | 32,259 | Operational pump. |
| `functional needs repair` | 4,317 | Operational pump requiring repair. |
| `non functional` | 22,824 | Pump is not operational. |

`id` is an identifier, so the 40 input columns comprise 39 potential predictors
and one identifier. Raw column types, missing counts, and cardinalities are
shown in [the audit notebook](../../notebooks/data_audit.ipynb). Counts and cleaning
decisions refer to the training dataset and are documented in
[the data quality findings](../../reports/sprint1/data_quality_findings.md).

## Raw Data Overview

Run the following code to view each column's data type, missing value count, and unique value count:

```python
rawdata_dictionary = pd.DataFrame({
    "Column": train_data.columns,
    "Data Type": train_data.dtypes.astype(str).values,
    "Missing Values": train_data.isnull().sum().values,
    "Unique Values": train_data.nunique().values,
})

rawdata_dictionary
```
