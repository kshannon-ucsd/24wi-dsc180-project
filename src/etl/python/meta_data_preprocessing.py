import numpy as np
import pandas as pd

def metadata_preprocessing(metadata, sepsis):
    metadata = metadata[metadata['subject_id'].notna()]

    metadata = metadata[[
        'subject_id',
        'charttime',
        'hadm_id',
        'stay_id',
        'heart_rate',
        'sbp',
        'sbp_ni',
        'mbp',
        'mbp_ni',
        'resp_rate',
        'temperature',
        'platelet',
        'wbc',
        'bands',
        'lactate',
        'inr',
        'ptt',
        'creatinine',
        'bilirubin',
        'pneumonia'
    ]]

    # getting most recent lab result
    recents = (metadata.sort_values(['subject_id', 'hadm_id', 'stay_id', 'charttime'])
            .groupby(['subject_id', 'hadm_id', 'stay_id'])
            .tail(1))

    recents = recents.reset_index().drop(columns='index')

    # use the most recent lab result as current data for each subject per admission
    means = (metadata.groupby(['subject_id', 'hadm_id', 'stay_id'])[['heart_rate',
                                                                'sbp',
                                                                'mbp',
                                                                'resp_rate',
                                                                'temperature',
                                                                'platelet',
                                                                'wbc',
                                                                'bands',
                                                                'lactate',
                                                                'inr',
                                                                'ptt',
                                                                'creatinine',
                                                                'bilirubin',
                                                                'pneumonia']]
            .mean()
            .reset_index())
    feat_squeeze = recents.combine_first(means)
    full_data = (sepsis.merge(feat_squeeze, how='left', on=['subject_id',
                                                        'hadm_id',
                                                        'stay_id'])
             .get(['heart_rate',
                   'sbp',
                   'mbp',
                   'resp_rate',
                   'temperature',
                   'platelet',
                   'wbc',
                   'bands',
                   'lactate',
                   'inr',
                   'ptt',
                   'creatinine',
                   'bilirubin',
                   'pneumonia',
                   'sepsis3']))

    full_data['sepsis3'] = np.where(full_data['sepsis3'] == 't', 1, 0)
    full_data['sepsis3'] = full_data['sepsis3'].astype(int)
    full_data = full_data.rename(columns={'sepsis3': 'sepsis'}, inplace=False)
    full_data.to_csv("data/processed/full_metadata.csv")
    
    return full_data