# Import packages
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing Functions
def date_format(date):
    """
    Takes in a date and formats it into the form YYYY-MM-DD

    Args:
        date (str): String that represents a date in the form of YYYYMMDD

    Returns:
        String that represents a date in the form of YYYY-MM-DD
    """
    date = str(date)
    formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    return formatted_date

def time_format(time):
    """
    Takes in a time and formats it into the form HH:MM:SS

    Args:
        time (str): String that represents time in the form of HH:MM:SS

    Returns:
        String that represents time in the form of HH:MM:SS
    """
    time = str(time)
    time = time.split(".")[0]
    # Ensure that all strings have the same format 
    while len(time) != 6:
        time = "0" + time
    formatted_time = f"{time[:2]}:{time[2:4]}:{time[4:6]}"
    return formatted_time

def convert_datetime(input_date):
    """
    Takes in date and time of the form YYYY-MM-DD HH:MM:SS and converts it into a datetime object

    Args:
        input_date (str): String that represents date and time in the form of YYYY-MM-DD HH:MM:SS

    Returns:
        Datetime object in iso format
    """
    return datetime.fromisoformat(input_date)

def main():
    # Loading in data
    subset = pd.read_csv("../data/external/subset.csv")


    #added pneumonia feature
    pneumonia = pd.read_csv("../data/external/second_features.csv")

    # Preprocessing data
    subset["admittime"] = subset["admittime"].apply(convert_datetime)
    subset["dischtime"] = subset["dischtime"].apply(convert_datetime)


    
    # complete_merged.to_csv("../data/processed/complete_merged.csv", index = False)


    #data preprocessing for model
    pneu = pneumonia[pneumonia['subject_id'].notna()]

    pneu = pneu[['subject_id', 'charttime','hadm_id', 'stay_id','heart_rate', 'sbp',
        'sbp_ni', 'mbp', 'mbp_ni', 'resp_rate', 'temperature', 'platelet',
        'wbc', 'bands', 'lactate', 'inr', 'ptt', 'creatinine', 'bilirubin', 'pneumonia']]
    recents = pneu.sort_values(['subject_id', 'hadm_id', 'stay_id', 'charttime']).groupby(['subject_id', 'hadm_id', 'stay_id']).tail(1)



    recents = recents.reset_index().drop(columns = 'index')
    recents.iloc[0]

    means = pneu.groupby(['subject_id', 'hadm_id', 'stay_id'])[['heart_rate', 'sbp',
        'sbp_ni', 'mbp', 'mbp_ni', 'resp_rate', 'temperature', 'platelet',
        'wbc', 'bands', 'lactate', 'inr', 'ptt', 'creatinine', 'bilirubin','pneumonia']].mean().reset_index()

    feat_squeeze = recents.combine_first(means)

    full_data = subset.merge(feat_squeeze, how = 'left', on = ['subject_id', 'hadm_id', 'stay_id'])


    full_data.to_csv("../data/processed/full_data.csv", index = False)




if __name__ == "__main__":
    main()