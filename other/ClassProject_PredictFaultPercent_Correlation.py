#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

Research_Area_Code = 42


# In[30]:


# Load Data
FaultPecent = pd.read_csv('data/Mon_Dis_JPCC_CRACK_INDEX.csv')
# TST = pd.read_csv('data/TST_L05B.csv')
Hum = pd.read_csv('data/humidity.csv')
Temp = pd.read_csv('data/CLM_VWS_TEMP_ANNUAL.csv')
Traffic = pd.read_csv('data/AnnualTraffic.csv')


# In[31]:


# TST = TST[TST.STATE_CODE == Research_Area_Code].drop(['SHRP_ID','STATE_CODE','STATE_CODE_EXP'], axis = 1)
Hum = Hum[Hum.STATE_CODE == Research_Area_Code].drop(['SHRP_ID','STATE_CODE','STATE_CODE_EXP'], axis = 1)
Temp = Temp[Temp.STATE_CODE == Research_Area_Code].drop(['SHRP_ID','STATE_CODE','STATE_CODE_EXP'], axis = 1)
Traffic = Traffic[Traffic.STATE_CODE == Research_Area_Code].drop(['SHRP_ID','STATE_CODE','STATE_CODE_EXP'], axis = 1)
FaultPecent = FaultPecent[FaultPecent.STATE_CODE == Research_Area_Code].drop(['SHRP_ID','STATE_CODE','STATE_CODE_EXP'], axis = 1)
# TST = TST[TST.STATE_CODE==Research_Area_Code][['CONSTRUCTION_NO','LAYER_NO','DESCRIPTION']]

FaultPecent = FaultPecent[['SURVEY_DATE','HPMS16_CRACKING_PERCENT_JPCC']]
# FaultPecent['SURVEY_DATE'] = pd.DatetimeIndex(FaultPecent['SURVEY_DATE']).year
# FaultPecent.rename(columns = {'SURVEY_DATE':'YEAR'}, inplace = True)
# TST = TST[['SHRP_ID','MATL_CODE']]
# Hum = Hum[['SHRP_ID','YEAR','MAX_ANN_HUM_AVG','MIN_ANN_HUM_AVG']]
# Temp = Temp[['SHRP_ID','YEAR','MEAN_ANN_TEMP_AVG']]
# Traffic = Traffic[['SHRP_ID','YEAR','ANNUAL_TRUCK_VOLUME_TREND']]


# FaultPecent = FaultPecent.dropna()
FaultPecent['SURVEY_DATE'] = pd.DatetimeIndex(FaultPecent['SURVEY_DATE']).year
FaultPecent.rename(columns = {'SURVEY_DATE':'YEAR'}, inplace = True)


# In[32]:


Weather = pd.merge(Temp,Hum, on = ['YEAR','VWS_ID']).drop(['VWS_ID'], axis = 1)
Prepared_data = pd.merge(FaultPecent,Weather, how = 'right',on = ['YEAR'])
Prepared_data = pd.merge(Prepared_data,Traffic,how = 'right', on = ['YEAR'])
Prepared_data = Prepared_data.dropna()

Target_Labels = Prepared_data['HPMS16_CRACKING_PERCENT_JPCC'].map(lambda x: x/100)
Prepared_data = Prepared_data.drop(['HPMS16_CRACKING_PERCENT_JPCC'], axis = 1)

from pandas.plotting import scatter_matrix
attributes = [col for col in Prepared_data.columns]
scat_matrix = scatter_matrix(Prepared_data[attributes], figsize=(27, 18))

for ax in scat_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 20, rotation = 90)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 20, rotation = 0)
