import pandas as pd
import pydicom as dicom
from skimage.transform import resize
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def knn(data):
    df = pd.read_csv(data)

    pixels = df['DicomPath'].apply(lambda x: resize(dicom.dcmread(x).pixel_array,(256,256)))

    X= np.stack(pixels)
    y = df['Abnormal']

    Xtrain = X[:70]
    ytrain = y[:70]
    Xtest = X[70:]
    ytest = y[70:]

    Xtrain_2d = Xtrain.reshape((15,256*256))

    nsamples, nx, ny = Xtest.shape
    Xtest_2d = Xtest.reshape((5,256*256))


    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(Xtrain_2d, ytrain) 
    y_pred = neigh.predict(Xtest_2d) 
    train_accuracy = accuracy_score(ytrain, y_pred)
    test_accuracy = accuracy_score(ytest, y_pred)
    return train_accuracy, test_accuracy










