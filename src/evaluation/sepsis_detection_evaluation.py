from sklearn.metrics import classification_report, accuracy_score

def catboost_classification_report(mdl, X_train, X_test, y_train, y_test):
    
    y_train_pred = mdl.predict(X_train)
    y_test_pred = mdl.predict(X_test)
    
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    print(classification_report(y_train, y_train_pred))
    print(accuracy_score(y_train, y_train_pred))
    
    print("==================", end='\n')
    
    print(classification_report(y_test, y_test_pred))
    print(accuracy_score(y_test, y_test_pred))
    
    return train_report, test_report