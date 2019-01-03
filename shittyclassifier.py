import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings 

warnings.filterwarnings('ignore')

# dataset is borrowed from Titanic competition from Kaggle
dataset_train = pd.read_csv('./titanic_train.csv')
dataset_test = pd.read_csv('./titanic_test.csv')
#print(dataset_train.head())

for column in dataset_train.columns:
    print(column)

dataset_train = dataset_train.dropna()
feature_columns = ['Sex'] # single feature is selected
target_column = ['Survived']
features = dataset_train[feature_columns].copy()
target = dataset_train[target_column].copy()
print(f'features.ndim: {features.ndim}')
print(f'target.ndim: {target.ndim}')

# Since working with strings are hard,
# strings are mapped into integer values
features = features.replace({'male':0, 'female':1})
print(f'Replacing male with 0 and female with 1')
#print(features)
#print(f'shape: {features.shape}')
#features = features.fillna(value=0)
#print(features)

print(f'final features frame shape: {features.shape}')
print(f'final features frame:\n{features.head()}')

print(f'final target frame shape: {target.shape}')
print(f'final target frame:\n{target.head()}')

# SVM based classifier
model = SVC()
print(f'Model details: {model}')

model.fit(features, target)

pred_male = model.predict(np.array(0, ndmin=2))
pred_female = model.predict(np.array(1, ndmin=2))
print(f'pred male: {pred_male}, female: {pred_female}')

print('***TESTING***')
dataset_test = dataset_test.dropna()
features_test = dataset_test[feature_columns].copy()
features_test = features_test.replace({'male':0, 'female':1})

print(f'Test features.ndim: {features_test.ndim}')
print(f'Test features.shape: {features_test.shape}')

pred = model.predict(features_test)
target_test = target[:len(pred)]
#print(f'pred:\n{pred}')
#print(f'target test:\n{target_test}')
score = accuracy_score(target_test, pred)
print(f'accuracy score: {score}')

