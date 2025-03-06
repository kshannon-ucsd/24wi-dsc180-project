#imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import classification_report

def second_model():
    full_data = pd.read_csv("../data/processed/full_data.csv")

    trial = full_data[['bilirubin', 'creatinine', 'heart_rate', 'inr', 'mbp', 'platelet',
       'ptt', 'resp_rate', 'sbp', 'wbc', 'pneumonia']]
    trial['sepsis'] = np.where(full_data.days == -1, 0, 1)




    X_abs = trial.dropna().drop(columns="sepsis")
    y_abs = trial.dropna().sepsis

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_abs, y_abs, test_size= 0.3, stratify=y_abs, random_state = 42)

    

    # Pipeline for classification
    positive_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(max_depth= 10, criterion='log_loss', n_estimators=80, \
                                            min_samples_split= 2, bootstrap=False, ccp_alpha = 0.0005, random_state=42))
    ])

    positive_pipeline.fit(X_a_train, y_a_train)

    y_train_pred = positive_pipeline.predict(X_a_train)
    y_test_pred =  positive_pipeline.predict(X_a_test)

    train_report = classification_report(y_a_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_a_test,y_test_pred, output_dict = True)

    return positive_pipeline

    


