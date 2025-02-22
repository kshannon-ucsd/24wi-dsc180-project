# Import packages
import pandas as pd
from datetime import datetime

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
    segmented = pd.read_csv("../data/external/CXLSeg-segmented.csv")
    xray = pd.read_csv("../data/external/mimic-cxr-2.0.0-metadata.csv")

    # Preprocessing data
    subset["admittime"] = subset["admittime"].apply(convert_datetime)
    subset["dischtime"] = subset["dischtime"].apply(convert_datetime)
    xray = xray.assign(formatted_date = xray["StudyDate"].apply(date_format))
    xray = xray.assign(formatted_time = xray["StudyTime"].apply(time_format))
    xray = xray.assign(studytime = (xray["formatted_date"] + " " + xray["formatted_time"]).apply(convert_datetime))

    # Merging
    # Subsetting xray dataset to make merge more efficient
    xray_merge = xray[["subject_id", "study_id", "ViewPosition", "studytime"]]
    # First merge
    merging = subset.merge(xray_merge, left_on = "subject_id", right_on = "subject_id")
    # Matching each xray to hospital admission
    matched_dates = merging[(merging["studytime"] >= merging["admittime"]) & (merging["studytime"] <= merging["dischtime"])].reset_index(drop = True)
    # Preprocessing segmented for merging
    segmented_merged = segmented[["subject_id", "study_id", "dicom_id", "DicomPath", "No Finding"]]
    segmented_merged["No Finding"] = segmented_merged["No Finding"].fillna(-1)
    segmented_merged["Abnormal"] = (segmented_merged["No Finding"] * -1)
    segmented_merged = segmented_merged.drop(columns = ["No Finding"])
    # Final merge
    complete_merged = matched_dates.merge(segmented_merged, on = ["subject_id", "study_id"])[["subject_id", "hadm_id", "stay_id", "study_id", 
                                                                        "admittime", "dischtime", "studytime", "ViewPosition",
                                                                        "dicom_id", "DicomPath", "Abnormal", "los", 
                                                                        "chronic_pulmonary_disease", "sepsis3"]]

    complete_merged.to_csv("../data/processed/complete_merged.csv", index = False)

if __name__ == "__main__":
    main()