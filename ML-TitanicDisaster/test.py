# process
# data analysis
import pandas as pd
import numpy as np
import time 
import re
import json
# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# ML library
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from keras.layers import Dropout,Dense
from sklearn.feature_selection import SelectFromModel



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline


#Acquire Training and Testing Data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
test_df_copy = test_df.copy()
df = [train_df, test_df]


#=========Ground-truth===========#
def create_groundtruth():
    test_data_with_labels = pd.read_csv('data/titanic.csv')
    test_data = pd.read_csv('data/test.csv')


    for i, name in enumerate(test_data_with_labels['name']):
        if '"' in name:
            test_data_with_labels['name'][i] = re.sub('"', '', name)
            
    for i, name in enumerate(test_data['Name']):
        if '"' in name:
            test_data['Name'][i] = re.sub('"', '', name)


    survived = []

    for name in test_data['Name']:
        survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))


    submission = pd.read_csv('data/gender_submission.csv')
    submission['Survived'] = survived
    submission.to_csv('data/groundtruth.csv', index=False)

def compare_with_groundtruth(pred):
    ground_truth = pd.read_csv('data/groundtruth.csv')
    precision = 0
    for i in range(len(pred)):
        if pred['Survived'][i] == ground_truth['Survived'][i]:
            precision +=1
        else:
            pass
    print('Ground truth accuracy: %s ' % np.round((precision/len(pred)),5))
    return np.round((precision/len(pred)),5)

for dataset in df:

	dataset['Surname'] = dataset['Name'].apply(lambda x: x.split(',')[0])

# New Ticket_id column
for dataset in df:
	dataset['Ticket_id'] = 'new_col'
	for i in range(len(dataset)):
	# Initialize Ticket_id = Pclass + Ticket + Fare + Embarked
		dataset['Ticket_id'][i] = str(dataset['Pclass'][i]) + '-' + str(dataset['Ticket'][i])[:-1] + '-' + str(dataset['Fare'][i]) + '-' + str(dataset['Embarked'][i])

# New Group_id column
for dataset in df:
	dataset['Group_id'] = 'new_col2'
	# Initialize Group_id = Surname + Ticket_id
	for i in range(len(dataset)):
	    dataset['Group_id'][i] = str(dataset['Surname'][i]) + '-' + str(dataset['Ticket_id'][i])

# creation of the Title feature
for dataset in df:
	dataset['Title'] = 'man'
	dataset.loc[dataset.Sex == 'female', 'Title'] = 'woman'
	dataset.loc[dataset['Name'].str.contains('Master'), 'Title'] = 'boy'


for dataset in df:
	dataset.loc[dataset.Title == 'man', 'Group_id'] = 'noGroup'

def remove_zero_fares(row):
    if row.Fare == 0:
        row.Fare = np.NaN
    return row
# Apply the function
for dataset in df:
    dataset = dataset.apply(remove_zero_fares, axis=1)
    # Check if it did the job
    print('Number of zero-Fares: {:d}'.format(dataset.loc[dataset.Fare==0].shape[0]))

# Calculate Ticket frequency and divide Fare by it
for dataset in df:
	dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
	dataset['Ticket_freq'] = dataset.groupby('Ticket')['Ticket'].transform('count')
	dataset['Pfare'] = dataset['Fare'] / dataset['Ticket_freq']
