import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as scipy
import time
import pandas as pd
import re
import multiprocessing
import validators 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
from statistics import mean


# Import dataset and drop unecessary columns
surface_temp_change = pd.read_csv("/Users/benjaminsionit/Downloads/STA141C-Project/Annual_Surface_Temperature_Change.csv").iloc[:,list(range(1, 4)) + list(range(10, 71))]
surface_temp_change = surface_temp_change.drop("ISO2", axis = "columns")
# Fix year values, rename columns, and get correct configuration of data
for i in range(2, 63):
    surface_temp_change.columns.values[i] = str(i + 1959)
surface_temp_change.rename(columns = {'Country':'Name', 'ISO3':'Country_code_A3'}, inplace = True)
surface_temp_change = surface_temp_change.melt(id_vars=["Name", "Country_code_A3"], var_name = "Year", 
        value_name = 'Surface Temperature Change')


# Import dataset and drop unecessary columns
land_cover = pd.read_csv("/Users/benjaminsionit/Downloads/STA141C-Project/Land_Cover_Accounts.csv").iloc[:,list(range(1, 4)) + list(range(11, 40))]
land_cover = land_cover.drop("ISO2", axis = "columns")
land_cover.rename(columns = {'Country':'Name', 'ISO3':'Country_code_A3'}, inplace = True)
# Fix year values, rename columns, and get correct configuration of data
for i in range(2, 31):
    land_cover.columns.values[i] = str(i + 1990)
land_cover = land_cover.melt(id_vars=["Name", "Country_code_A3"], var_name = "Year", value_name = 'Land Coverage')

# Merge data sets and drop any null values
x = pd.merge(land_cover, surface_temp_change, how = "left", on=["Name", "Country_code_A3", "Year"]).dropna()

# Import dataset and drop unecessary columns
food_emis = pd.read_excel("/Users/benjaminsionit/Downloads/STA141C-Project/EDGAR-FOOD_v61_AP.xlsx", sheet_name = 3, header = None).iloc[3:]
food_emis.columns = food_emis.iloc[0]
food_emis = food_emis.iloc[1:]
# Fix year values, rename columns, and get correct configuration of data
for i in range(3, 52):
    food_emis.columns.values[i] = str(i + 1967)
food_emis = food_emis.pivot(index = ["Name", "Country_code_A3"], columns = "Substance").reset_index()
food_emis = food_emis.melt(id_vars=["Name", "Country_code_A3"], var_name = ["Year", "Substance"], value_name = "Sub")
food_emis = food_emis.pivot(index = ["Name", "Country_code_A3", "Year"], columns = "Substance").reset_index()
food_emis.columns = food_emis.columns.droplevel()
food_emis.columns = ["Name", "Country_code_A3", "Year", "BC", "CO", "NH3", "NMVOC", "NOx", "OC", "PM10", "PM2.5", "SO2"]

# Import dataset and drop unecessary columns
co2 = pd.read_excel("/Users/benjaminsionit/Downloads/STA141C-Project/IEA_EDGAR_CO2_1970-2021.xlsx", sheet_name = 2).iloc[9:, 2:]
co2.columns = co2.iloc[0]
co2 = co2.iloc[1:]
# Fix year values, rename columns, and get correct configuration of data
co2 = co2.melt(id_vars=["Name", "Country_code_A3", "Substance"], var_name = "Year", value_name = "CO2")
co2.Year = co2.Year.apply(lambda x: pd.Series(str(x).split("_")[1]))
co2 = co2.drop("Substance", axis = 1)

# Import dataset and drop unecessary columns
ch4 = pd.read_excel("/Users/benjaminsionit/Downloads/STA141C-Project/EDGAR_CH4_1970-2021.xlsx", sheet_name = 2).iloc[9:, 2:]
ch4.columns = ch4.iloc[0]
ch4 = ch4.iloc[1:]
# Fix year values, rename columns, and get correct configuration of data
ch4 = ch4.melt(id_vars=["Name", "Country_code_A3", "Substance"], var_name = "Year", value_name = "CH4")
ch4.Year = ch4.Year.apply(lambda x: pd.Series(str(x).split("_")[1]))
ch4 = ch4.drop("Substance", axis = 1)

# Import dataset and drop unecessary columns
n2o = pd.read_excel("/Users/benjaminsionit/Downloads/STA141C-Project/EDGAR_N2O_1970-2021.xlsx", sheet_name = 2).iloc[9:, 2:]
n2o.columns = n2o.iloc[0]
n2o = n2o.iloc[1:]
# Fix year values, rename columns, and get correct configuration of data
n2o = n2o.melt(id_vars=["Name", "Country_code_A3", "Substance"], var_name = "Year", value_name = "N2O")
n2o.Year = n2o.Year.apply(lambda x: pd.Series(str(x).split("_")[1]))
n2o = n2o.drop("Substance", axis = 1)

# Merge data sets
#y = pd.merge(co2, ch4, how = "left", on=["Name", "Country_code_A3", "Year"])
#y = pd.merge(y, n2o, how = "left", on=["Name", "Country_code_A3", "Year"])
#y = pd.merge(food_emis, y, how = "left", on=["Name", "Country_code_A3", "Year"])

# Merge data sets, drop any null values, and convert appropriate columns to be numeric
final = pd.merge(x, food_emis, how = "left", on=["Name", "Country_code_A3", "Year"]).dropna()
final = final.drop(["OC", "NMVOC"], axis = 1).dropna()
final["Year"] = pd.to_numeric(final["Year"])
#final["CO2"], final["CH4"], final["N2O"] = pd.to_numeric(final["CO2"]), pd.to_numeric(final["CH4"]), pd.to_numeric(final["N2O"])
# Re-define subregions according to classification
def UN_subregion(region):
    if ((region == 'Northern Europe') | (region == 'Southern Europe') | (region == 'Eastern Europe') | (region == 'Western Europe')):
        region = "Europe"
    elif((region == "Southern Asia") | (region == "Central Asia") | (region == "Western Asia")):
        region = "South-Central Asia"
    elif ((region == "South-eastern Asia") | (region == "Eastern Asia") | (region == "Melanesia") | (region == "Micronesia") |
         (region == "Polynesia")):
        region = "South-East Asia and Polynesia"
    return(region)
# Import subregion data, update values according to classification
UN = pd.read_csv("/Users/benjaminsionit/Downloads/UNCODES.csv")
UN["sub-region"] = UN["sub-region"].apply(UN_subregion)
UN = UN.iloc[:, [2, 6]]
UN.rename(columns = {'alpha-3':'Country_code_A3', 'sub-region':'Region'}, inplace = True)
final = pd.merge(final, UN, how = "left", on=["Country_code_A3"]).dropna().iloc[:, 2:]
# Add constant and convert categorical variables to dummy variables
final = sm.add_constant(final)
final = pd.get_dummies(data = final, drop_first = True)
final.to_csv("data.csv")

# Define predictor and regressors
y = final["Surface Temperature Change"]
x = final.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
X_variables = x




def model_fit_calculator(y, x):
    (n,k) = x.shape
    model = sm.OLS(list(y),x).fit()
    model.fittedvalues
    fitted_y = model.fittedvalues
    mean_y = mean(y)
    SSR = sum((fitted_y-mean(y))**2)
    SSE = sum((model.resid)**2)
    SST = SSR + SSE
    LogL = model.llf
    AIC = 2*k - 2*LogL
    BIC = k*np.log(n) - 2*LogL
    R_squared = 1 - (SSE/SST)
    VIF = 1/(1 - R_squared)
    return(pd.Series({"Dependent": 'Surface Temperature Change', "Independent": list(x.columns)[0:], "AIC": AIC, "BIC": BIC, "R-Squared": R_squared, "VIF": VIF}))


def Get_Combinatorial(comb):
    combinatoric_model = X_variables.iloc[:, comb]
    return(model_fit_calculator(y, combinatoric_model).to_frame().T)



def Get_List(critlist):
    model_info = pd.DataFrame(columns = ["Dependent", "Independent", "AIC", "BIC", "R-Squared", "VIF"])
    for i in range(1,len(critlist)):
        model_info = pd.concat([model_info,critlist[i]], ignore_index = True)
    return(model_info)

    
    

