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
| `funder` | Who funded the well. |
| `gps_height` | Altitude of the well. |
| `installer` | Organization that installed the well. |
| `longitude` | GPS longitude coordinate. |
| `latitude` | GPS latitude coordinate. |
| `wpt_name` | Name of the waterpoint, if available. |
| `num_private` | Not defined in the source description; provides no information. |
| `basin` | Geographic water basin. |
| `subvillage` | Geographic location. |
| `region` | Geographic location. |
| `region_code` | Geographic location (coded). |
| `district_code` | Geographic location (coded). |
| `lga` | Geographic location. |
| `ward` | Geographic location. |
| `population` | Population around the well. |
| `public_meeting` | True/false indicator. |
| `recorded_by` | Group entering the row of data. |
| `scheme_management` | Who operates the waterpoint. |
| `scheme_name` | Who operates the waterpoint. |
| `permit` | Whether the waterpoint is permitted. |
| `construction_year` | Year the waterpoint was constructed. |
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
