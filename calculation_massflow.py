# Functions used to calculate the mass flow through the door

import pandas as pd
import statsmodels.api as sm
import numpy as np

def calculation_area(number_of_heights = 9, delta_height = 0.2, door_width = 0.8):
    """
    Returns a list with the equivalent area fraction of the door for each probe.
    
    The list provides a number of areas equal to the number of heights measured. Some heights (0.4 m and 1.6 m) have three pressure probes.
    
    Parameters:
    ----------
    number_of_heights: number of heights at which pressure probes were located
        int
    
    delta_height: fraction of the door corresponding to a single pressure probe
        float
        
    door_width: width of the door
        float
        
    Returns:
    -------
    areas = list with the corresponding door area for each pressure probe height
        list
    """
    areas = [delta_height * door_width] * number_of_heights
    
    return areas


def calculation_velocity(df, gamma = 0.94, omega_factor = 2.49, gems_factor = 10, lowess_fraction = 0.05):
    """
    Calculates gas flow velocity from the pressure probe readings
    
    Parameters:
    ----------
    df: pandas DataFrame with the raw test data
        pd.DataFrame
        
    gamma: calibration constant for the pressure probe
        float
        
    omega_factor: conversion factor for the omega pressure transducers
        float
        
    gems_factor: conversion factor for the gems pressure transducers
        int
        
    lowess_fraction: fraction of the data used when estimating each smoothed value
        float
        
    Returns:
    -------
    df: pandas DataFrame with the formatted and calculated data
        pd.DataFrame
    """
    # create a mask to access values before start test
    mask_prestart = df.loc[:, "Time [min]"] < 0
    
    # calculate ambient temperature as mean value of all temperatures before start and assing to h = 20 cm.
    temperature_columns = []
    for column in df:
        if "TDD" in column:
            temperature_columns.append(column)
    temperature_ambient = df.loc[mask_prestart, temperature_columns].mean().mean()
    df["TDD.20"] = temperature_ambient
    
    pressure_columns = []
    for column in df:
        if "PP" in column:
            pressure_columns.append(column)
    pressure_prestart_mean = df.loc[mask_prestart, pressure_columns].mean()
    
    # calculate the zeroed values for the pressrue channels (value - mean_prestart)
    for column in pressure_prestart_mean.index:
        df.loc[:, f"{column}_zeroed"] = df.loc[:, column] - pressure_prestart_mean[column]

    # determine which pressure transducers were used for each sensor
    omega = [f"PP0{x}" for x in range(1,7)]
    gems = [f"PP0{x}" for x in range(7,10)] + [f"PP{x}" for x in range(10,14)]
    
    # calculate the pressure difference from the zeroed values
    for column in df:
        if "zeroed" not in column:
            pass
        else:
            probe = column.split("_")[0]
            if probe in omega:
                df.loc[:, f"{probe}_DeltaP"] = df.loc[:, column] * omega_factor
            elif probe in gems:
                df.loc[:, f"{probe}_DeltaP"] = df.loc[:, column] * gems_factor
    
    # smooth the pressure readings using a lowess algorithm
    lowess = sm.nonparametric.lowess
    for column in df:
        if "DeltaP" in column:
            df.loc[:, f"{column}_smooth"] = lowess(df.loc[:, column], df.loc[:, "Time [min]"],
                   frac = lowess_fraction, return_sorted = False)
    
    # drop all other pressure columns (except for pressure smooth)
    PP_drop = []
    for column in df:
        if ("PP" in column) and not ("smooth" in column):
            PP_drop.append(column)
    df1 = df.drop(columns = PP_drop)
    
    # average those probes which are at the same height
    df1.loc[:, "PP_40"] = df1.loc[:, ["PP02_DeltaP_smooth", "PP03_DeltaP_smooth", "PP04_DeltaP_smooth"]].mean(axis = 1)
    df1.loc[:, "PP_160"] = df1.loc[:, ["PP10_DeltaP_smooth", "PP11_DeltaP_smooth", "PP12_DeltaP_smooth"]].mean(axis = 1)
    
    df1.drop(columns = ["PP02_DeltaP_smooth", "PP03_DeltaP_smooth", "PP04_DeltaP_smooth", 
                        "PP10_DeltaP_smooth", "PP11_DeltaP_smooth", "PP12_DeltaP_smooth"], inplace = True)
    
    # rename columns to their respective height
    df1.rename(columns = {"PP01_DeltaP_smooth": "PP_20", "PP05_DeltaP_smooth": "PP_60", 
                        "PP06_DeltaP_smooth": "PP_80", "PP07_DeltaP_smooth": "PP_100", 
                        "PP08_DeltaP_smooth": "PP_120", "PP09_DeltaP_smooth": "PP_140", 
                        "PP13_DeltaP_smooth": "PP_180"}, inplace = True)
    
    # for a given height, if delta p is positive then temperature equals ambient temperature
    for column in df1:
        if "PP" in column:
            mask_positives = df1.loc[:, column] > 0
            height = column.split("_")[1]
            df1.loc[:, f"TC_{height}"] = df1.loc[:, f"TDD.{height}"]
            df1.loc[mask_positives, f"TC_{height}"] = temperature_ambient
            
    # drop the old temperature columns
    df1.drop(columns = [f"TDD.{x}" for x in range(20,200,20)], inplace = True)
    
    # calculate density
    for column in df1:
        if "TC" in column:
            height = column.split("_")[1]
            df1.loc[:, f"Rho_{height}"] = 353 / (df1.loc[:,column] + 273)
    
    # calculate velocities
    for height in range(20,200,20):
        df1.loc[:,f"V_{height}"] = gamma * np.sqrt(2 * np.abs(df1.loc[:, f"PP_{height}"]) / 
               df1.loc[:, f"Rho_{height}"])
        
        # np.abs used for calculation but now negative delta P should give a negative (outward) velocity
        mask_negatives = df1.loc[:, f"PP_{height}"] < 0
        df1.loc[mask_negatives,f"V_{height}"] = df1.loc[mask_negatives,f"V_{height}"] * -1
        
    return df1

def calculation_massflow(df, areas, Cd = 0.68):
    """
    Calculates the mass flow from the velocities and areas already determined.
    
    Also determines the total inflow and outflow of gases to and from the compartment
    
    Parameters:
    ----------
    df: pandas DataFrame containing all the time dependant data
        pd.DataFrame
    
    areas: fraction of the door area to which each pressure probe corresponds
        list
        
    Cd: discharge coefficient (0.68 according to SFPE and 0.7 according to Prahl and Emmons, 1975)
    
    Returns:
    -------
    df: pandas DataFrame containing all the data.
    """
    
    for i, height in enumerate(list(range(20,200,20))):
        df.loc[:, f"M_{height}"] = Cd * df.loc[:, f"Rho_{height}"] * df.loc[:,f"V_{height}"] * areas[i]
    
    mass_columns = []
    for column in df:
        if "M_" in column:
            mass_columns.append(column)
            
    # sum positives and negatives to obtain mass_in and mass_out (slow and dirty implementation)
    for index, row in df.loc[:, mass_columns].iterrows():
        positives = 0
        negatives = 0
        print(index,row)
        for item in row:
            if item > 0:
                positives += item
            else:
                negatives =+ np.abs(item)
        df.loc[index, "mass_in"] = positives
        df.loc[index, "mass_out"] = negatives
            
    return None

def calculation_HRR(df, XO2_0 = 0.2095, alpha = 1.105, E_02 = 13.1, ECO_CO2 = 17.6, E_CO2 = 13.3, 
                    E_CO = 12.3, M_a = 29, M_O2 = 32, M_CO2 = 44, M_CO = 28):
    """
    Calculates the HRR from the a
    
    """
    
    return None
