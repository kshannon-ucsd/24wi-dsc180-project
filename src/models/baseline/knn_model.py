from sklearn.neighbors import KNeighborsClassifier


def train_knn_model(X_train, y_train, n_neighbors=3):
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
    
    knn.fit(X_train, y_train)
    
    return knn
