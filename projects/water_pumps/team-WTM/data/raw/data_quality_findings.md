Findings Summary from the Data Quality Checks

Dataset contains 41 columns and 59400 rows

1. The feature names and data types have been documented and can be found in the data_dictionary and the notebook

2. The features with missing values are:

scheme_name              28810
scheme_management         3878
installer                 3655
funder                    3637
public_meeting            3334
permit                    3056
subvillage                 371
wpt_name                     2

3. Also some of the categorial variables have related information so one can be kept.

Features that are very related:

Funder              → installer

source              → source_type

quantity            → quantity_group

water_quality       → quality_group

extraction_type     → extraction_type_group 

waterpoint_type     → waterpoint_type_group

management           → management_group

4. The datat_recorded is string so this should be coverted to datatime type


5. The high-cardinal feature variables have been identified can be found in the notebook
wpt_name, subvillage, scheme_name, installer, ward, funder, lga, region ......

6. For the numerical variables
No missing values in the numeric_columns

#num_private column can be removed because it identify invalid  values: mostly zeros

population and construction year should be investigated, as their are 0 values whicha are invalid and showuld be handled differently. We can replace with missing or unknown

gps_height can be zero or negative

If zero it means the waterpoint is located approximately at sea level

Negative would mean the location is below sea level.


