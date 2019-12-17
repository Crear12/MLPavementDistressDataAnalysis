# MLPavementDistressDataAnalysis
It's a class project using Machine Learning for analysis of pavement distress data downloaded from LTPP InfoPave. In this project, construction, annual humidity, temperature and traffic data form LTPP InfoPave was adopted to predict concrete pavement cracking percentage. Linear regression model, decision tree model and random forests model were used to predict and compared with each other. Random forest has advantage in accuracy and decision tree has advantage in speed. Both models did not show their best performance due to lack of effective data.

# Acknowledgments
The project team would like to thank U.S. Department of Transportation Federal Highway Administration and LTPP InfoPave for the pavement distress data, traffic data and climate data. The project team would also like to thank Pengyu Xie for his instruction on pavement distress data feature selection.

# Data Sources
## Website-based Database
https://infopave.fhwa.dot.gov/Data/DataSelection

## Data Resolution
At 1 year interval.

## Code Environment
IDE: Python Jupyter Notebook.
Modules: Pandas,Numpy,Sklearn,Matplotlib,tqdm

## Feature Selection
SHRP_ID, STATE_CODE, YEAR, SURVEY_DATE, HPMS16_CRARKING_PERCENT_JPCC, MEAN_ANN_ TEMP_AVG, FREEZE_INDEX_YR, FREEZE_THAW_YR, MAX_ANN_HUM_AVG, MIN_ANN_ HUM_AVG, CONSTRUCTION_NO, AADTT_ALL_TRUCKS_TREND, ANNUAL_TRUCK_ VOLUME_TREND and REPR_THICKNESS.

## Merging Logic
Firstly, the humidity and temperature data are merged by 'SHRP_ID', 'STATE_CODE', 'YEAR', 'VWS_ID'. Secondly, merge it with cracking data and traffic data by 'SHRP_ID', 'STATE_CODE', 'YEAR' one by one. At last, merge it with construction data by 'SHRP_ID', 'STATE_CODE', 'CONSTRUCTION_NO'. Considering that the target is JPCC cracking percentage, only the concrete construction data were kept.

'SHRP_ID' is test section identification number assigned by LTPP program. Must be combined with STATE_CODE to be unique.
'STATE_CODE' is numerical code for state or province. U.S. codes are consistent with Federal Information Processing Standards.

'VWS_ID' is code that uniquely identifies virtual weather station.

'YEAR' is the by year temporal information of the data. The cracking data provides ‘SURVEY_DATE’ rather than ‘YEAR’. To match it with other data, the date information has to be transformed into year.

'CONSTRUCTION_NO' is event number used to relate changes in pavement structure with other time dependent data elements. This field is set to 1 when a test section is initially accepted into LTPP and is incremented with each change to the layer structure.

The principle for data merging is to find the unique columns as indices. The indices chosen in this project are time and location.

# Machine Learning
## Models
Linear Regression, Decision Tree, Random Forests.

## Figures
Prediction Results, Learning Curves
