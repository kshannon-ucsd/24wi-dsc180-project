from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def second_model_baseline(X_train, y_train):
    
    params = {
        'n_estimators': 80,         
        'max_depth': 2,                 
        'min_samples_split': 2,     
        'criterion': 'log_loss',         
        'random_seed': 42,           
        'bootstrap': False,
        'ccp_alpha': 0.0005,              
    }
    
    # Pipeline for classification
    mdl = Pipeline([
        ('classifier', RandomForestClassifier(**params))
    ])

    mdl.fit(X_train, y_train)
    
    return mdl