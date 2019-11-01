# this analysis is based on the file sent by Juan on an email dated 11/08/2016 -(BRE experiments, 2016)

import pandas as pd 

# import my own functions
from calculation_massflow import calculation_area, calculation_velocity, calculation_massflow


experiments = ["Alpha2", "Beta2", "Gamma"]
experiments = ["Gamma"]
for experiment in experiments:
    df = pd.read_excel("Data_DoorAnalysis_Alltests.xlsx", sheet_name = experiment)
    
    # calculate areas
    areas = calculation_area()
    
    # calculate velocities
    df = calculation_velocity(df)
    
    # calculate mass flow
    calculation_massflow(df, areas)
    
    # calculate HRR
    
    # save data frame
    df.to_pickle(f"doordata_{experiment}")



