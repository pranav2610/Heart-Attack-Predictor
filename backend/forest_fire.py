#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

dataset = pd.read_csv('cardio_train_new_upload_1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=33)
knn.fit(X_train, y_train)

pickle.dump(knn, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

