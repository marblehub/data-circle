Data Description 

    This is water pump dataset sourced from Tanazania and the goal is to predict the functionality status of the waterpoint.

    The Training_set_values contains information about each waterpoint that can potentially help explain whether the pump is functioning or not.

    The training_set_labels contains the traget variable that we ultimately want to predict.

    The id column connects each label to the corresponding waterpoint.

    Both dataset set as both the id columns, this makes merging easy.

    The two dataset will be merged so that all information about each waterpoint and its corrreponding functionality status are contained in one dataset. This makes it easier to perform data quality checks and exploratory analysis.

## The merged dataset
The merged dataset contains 41 columns and 59400 rows. Each row represents a waterpoint, while the columns contain information about its location, construction, management, water source, water quality, population and functionality status. 
The dataset contains both numerical and categorical variables.

We have 1 Target variable and 40 feature variables all defined below: 

amount_tsh - Total static head (amount water available to waterpoint)
date_recorded - The date the row was entered
funder - Who funded the well
gps_height - Altitude of the well
installer - Organization that installed the well
longitude - GPS coordinate
latitude - GPS coordinate
wpt_name - Name of the waterpoint if there is one
num_private -
basin - Geographic water basin
subvillage - Geographic location
region - Geographic location
region_code - Geographic location (coded)
district_code - Geographic location (coded)
lga - Geographic location
ward - Geographic location
population - Population around the well
public_meeting - True/False
recorded_by - Group entering this row of data
scheme_management - Who operates the waterpoint
scheme_name - Who operates the waterpoint
permit - If the waterpoint is permitted
construction_year - Year the waterpoint was constructed
extraction_type - The kind of extraction the waterpoint uses
extraction_type_group - The kind of extraction the waterpoint uses
extraction_type_class - The kind of extraction the waterpoint uses
management - How the waterpoint is managed
management_group - How the waterpoint is managed
payment - What the water costs
payment_type - What the water costs
water_quality - The quality of the water
quality_group - The quality of the water
quantity - The quantity of water
quantity_group - The quantity of water
source - The source of the water
source_type - The source of the water
source_class - The source of the water
waterpoint_type - The kind of waterpoint
waterpoint_type_group - The kind of waterpoint
status_group - defines the functionality status of the water pumps


To get a better overview of the raw data run this code. it shows the datatypes, unique values and the missing values 
"
rawdata_dictionary = pd.DataFrame({
    "Column": train_data.columns,
    "Data Type": train_data.dtypes.astype(str).values,
    "Missing Values": train_data.isnull().sum().values,
    "Unique Values": train_data.nunique().values
})
rawdata_dictionary
"

