import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
from sklearn.metrics import precision_recall_curve


def preprocess_data(mimic, feats, segm, metadata):
    subset = pd.read_csv(mimic)
    features = pd.read_csv(feats)

    segmented = pd.read_csv(segm)
    xray = pd.read_csv(metadata)

    def date_format(date):
        date = str(date)
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        return formatted_date

    def time_format(time):
        time = str(time).split(".")[0]
        while len(time) != 6:
            time = "0" + time
        formatted_time = f"{time[:2]}:{time[2:4]}:{time[4:6]}"
        return formatted_time

    def convert_datetime(input_date):
        return datetime.fromisoformat(input_date)
    
    subset["admittime"] = pd.to_datetime(subset["admittime"])
    subset["dischtime"] = pd.to_datetime(subset["dischtime"])
    subset['suspected_infection_time'] = pd.to_datetime(subset['suspected_infection_time'])

    xray = xray.assign(formatted_date = xray["StudyDate"].apply(date_format))
    xray = xray.assign(formatted_time = xray["StudyTime"].apply(time_format))
    xray = xray.assign(studytime = (xray["formatted_date"] + " " + xray["formatted_time"]).apply(convert_datetime))

    subset = subset[\
    (((subset['suspected_infection_time'].dt.normalize()-subset['admittime'].dt.normalize()).dt.days)>=0)\
    | (((subset['suspected_infection_time'].dt.normalize()-subset['admittime'].dt.normalize()).dt.days).isna())
    ]

    subset['days'] = (subset['suspected_infection_time'].dt.normalize()-subset['admittime'].dt.normalize()).dt.days
    subset['days'] = subset['days'].fillna(-1)

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
                                                                        "admittime", "dischtime", "days", "studytime", "ViewPosition",
                                                                        "dicom_id", "DicomPath", "Abnormal", "los", 
                                                                        "chronic_pulmonary_disease", "sepsis3"]]

    features = features[features['subject_id'].notna()]
    recents = features.sort_values(['subject_id', 'hadm_id', 'stay_id', 'charttime']).groupby(['subject_id', 'hadm_id', 'stay_id']).tail(1)

    recents = recents.reset_index().drop(columns = 'index')

    means = features.groupby(['subject_id', 'hadm_id', 'stay_id'])[['heart_rate', 'sbp',
       'sbp_ni', 'mbp', 'mbp_ni', 'resp_rate', 'temperature', 'platelet',
       'wbc', 'bands', 'lactate', 'inr', 'ptt', 'creatinine', 'bilirubin']].mean().reset_index()
    
    feat_squeeze = recents.combine_first(means)
    full_data = complete_merged.merge(feat_squeeze, how = 'left', on = ['subject_id', 'hadm_id', 'stay_id'])
    full_data = full_data.drop_duplicates('dicom_id')

    feats = ['Abnormal', 'bilirubin', 'creatinine', 'heart_rate', 'inr', 'mbp', 'platelet',
       'ptt', 'resp_rate', 'sbp', 'wbc', 'days']

    X = full_data[feats].drop_duplicates()


    for col in X.columns[1:-1]:
        if X[col].isnull().sum() > 0:  # Check if column has missing values
                sampled_values = X[col].dropna().sample(X[col].isnull().sum(), replace=True, random_state=42).values
                X.loc[X[col].isnull(), col] = sampled_values
    
    scaler = StandardScaler()
    scale_feats = X.drop(columns = ['days', 'Abnormal']).columns
    X_scaled = X.copy()
    X_scaled[scale_feats] = scaler.fit_transform(X[scale_feats])
    X_scaled = X_scaled.drop(columns = 'days')

    # Convert target labels to binary format (multilabel)
    y = pd.get_dummies(full_data[feats].drop_duplicates()['days'].apply(lambda x: '2+' if x >=2 else str(x)))

    y = y.idxmax(axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def build_model(X_train, X_test, y_train, y_test):

    # Apply SMOTE to Training Data Only
    smote = SMOTE(sampling_strategy='auto', random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    rf_clf = RandomForestClassifier(random_state = 2, max_depth = 10, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 100, class_weight='balanced', criterion='entropy')
    rf_clf.fit(X_train_resampled, y_train_resampled)

    classification_report_train = classification_report(y_train, rf_clf.predict(X_train))
    classification_report_test = classification_report(y_test, rf_clf.predict(X_test))

    train_accuracy = accuracy_score(y_train, rf_clf.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))

    confusion_matrix = confusion_matrix(y_test, rf_clf.predict(X_test))
    y_prob = rf_clf.predict_proba(X_test)

    return classification_report_train, classification_report_test, train_accuracy, test_accuracy, confusion_matrix, y_prob

def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(base_dir, 'output', 'mimic_dataset', 'baseline')

    subset = os.path.join(base_dir, 'data', 'preprocessed', 'second_model_subset.csv')
    features = os.path.join(base_dir, 'data', 'preprocessed', 'features.csv')

    segmented = os.path.join(base_dir, 'data', 'preprocessed', 'CXLSeg-segmented.csv')

    xray = os.path.join(base_dir, 'data', 'preprocessed', 'CXLSeg-metadata.csv')

    X_train, y_train, X_test, y_test = preprocess_data(subset, features, segmented, xray)
    class_report_train, class_report_test, train_acc, test_acc, confusion_matrix, y_prob = \
        build_model(X_train, y_train, X_test, y_test)


    results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'class_report_train': class_report_train,
            'class_report_test': class_report_test,
            'confusion_matrix': confusion_matrix.tolist()
    }

    print('Train Accuracy: '+ train_acc)
    print('Test Accuracy: ' + test_acc)

    print(class_report_train)
    print(class_report_test)


    with open(os.path.join(output_dir, 'model2_baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    #roc plot
    class_labels = ['Class -1', 'Class 0', 'Class 1', 'Class 2+']

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(y_test_bin.shape[1]):  
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])  
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(y_test_bin.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})') 

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  
    plt.title('ROC Curve for Sepsis Prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    plt.savefig(output_dir)
    plt.close()

if __name__ == '__main__':
    main()


    








     
