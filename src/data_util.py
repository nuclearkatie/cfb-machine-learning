import numpy as np
import pandas as pd
import csv
import argparse


def load_clean(filename, columns):

    data = pd.read_csv(filename, sep=',',header=0)
    data = clean_data(data, columns)
    data = pop_winner(data)

    return data


def clean_data(data, columns):

    #drop user-requested columns
    data = data.drop(columns = columns)

    #drop scores if they exist to ensure they don't end up in the final data set
    try:
        data = data.drop(columns = ["Home_score", "Away_score"])
        print("Game scores found. Dropping them so they don't affect classifier \n Data imported successfully")
    except:
        print("No scores found. Data imported successfully")

    return data


def standard_dropped_columns():

    columns = ["Unnamed: 0",
               "H_Poll_curr", "H_Poll_pre", "H_Poll_high","A_Poll_curr",
               "A_Poll_pre", "A_Poll_high",
               'H_Overall_Pct', 'H_Conf_Pct', 'H_Offensive_Pass_Pct', 'H_Offensive_Rush_Yards_Average',
                'H_Offensive_Total_Average', 'H_Offensive_First_Downs_Total', 'H_Offensive_Turnovers_Total',
                'H_Defensive_Pass_Pct', 'H_Defensive_Rush_Yards_Average', 'H_Defensive_Total_Average',
                'H_Defensive_First_Downs_Total', 'H_Defensive_Turnovers_Total',
                'H_Offensive_Total_Plays', 'H_Offensive_Total_Yards',
                'H_Defensive_Total_Plays', 'H_Defensive_Total_Yards',
                'A_Overall_Pct', 'A_Conf_Pct', 'A_Offensive_Pass_Pct', 'A_Offensive_Rush_Yards_Average',
                'A_Offensive_Total_Average', 'A_Offensive_First_Downs_Total', 'A_Offensive_Turnovers_Total',
                'A_Defensive_Pass_Pct', 'A_Defensive_Rush_Yards_Average', 'A_Defensive_Total_Average',
                'A_Defensive_First_Downs_Total', 'A_Defensive_Turnovers_Total',
                'A_Offensive_Total_Plays', 'A_Offensive_Total_Yards',
                'A_Defensive_Total_Plays', 'A_Defensive_Total_Yards']

    return columns


def pop_winner(data):

    try:
        cols = list(data.columns.values) #Make a list of all of the columns in the df
        cols.pop(cols.index('Winner')) #Remove b from list
        data = data[cols+['Winner']] #Create new dataframe with columns in the order you want
    except:
        return data

    return data
