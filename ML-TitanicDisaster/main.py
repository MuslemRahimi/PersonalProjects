# website
#https://www.kaggle.com/code/shilongzhuang/attack-on-titanic-solution-walkthrough-top-8

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
from keras.layers import Dropout
from sklearn.feature_selection import SelectFromModel



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
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


# ratio of test/train
#print(len(test_df)/(len(train_df)+len(test_df)))


#Add feature: Family Size
for dataset in df:
    dataset['Fam_size'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Fam_type'] = pd.cut(dataset.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])


for dataset in df:
    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

for dataset in df:
    dataset['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
    dataset['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)


for dataset in df:
    dataset['Age_type'] = pd.cut(dataset['Age'], [8,16,32,64,100], labels=['Child', 'Teen', 'Adult', 'Senior'])


for dataset in df:
    # Extract the first two letters
    dataset['Ticket_lett'] = dataset.Ticket.apply(lambda x: x[:2])
    # Calculate ticket length
    dataset['Ticket_len'] = dataset.Ticket.apply(lambda x: len(x))





def remove_zero_fares(row):
    if row.Fare == 0:
        row.Fare = np.NaN
    return row

# Apply the function
for dataset in df:
    dataset = dataset.apply(remove_zero_fares, axis=1)
    # Check if it did the job
    print('Number of zero-Fares: {:d}'.format(dataset.loc[dataset.Fare==0].shape[0]))



drop_columns = ['PassengerId','Name', \
               'Cabin', 'Age',\
               'Ticket','SibSp','Parch','Fam_size','Age_type']

#Since "Sex" it's a binary variable, we don't need to keep a feature 
#for male and for female

# Dropping irrelevant column features

train_df.drop(drop_columns, inplace=True, axis=1)
test_df.drop(drop_columns, inplace=True, axis=1)
df = [train_df, test_df]


numerical_cols = ['Fare']
categorical_cols = ['Pclass', 'Title', 'Embarked', 'Fam_type',\
                   'Sex','Ticket_len','Ticket_lett']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


def find_best_para(model_name,preprocessor):
    if model_name == 'clf':
        model = RandomForestClassifier()
        param_grid = {'model__criterion':['entropy','gini','log_loss'],\
                      'model__n_estimators': [50,100,150,200,500,600],\
                      'model__min_samples_split':[1,3,5,7.9],\
                      'model__min_samples_leaf': [1,3,5,7,9],\
                      'model__max_depth': [1,3,5,7,9,10]}
    elif model_name == 'dt':
        model = DecisionTreeClassifier()
        param_grid = {'model__criterion':['entropy','gini','log_loss'],\
                      'model__min_samples_split':[1,3,5,7],\
                      'model__min_samples_leaf': [1,3,5,7],\
                      'model__max_depth': [1,3,5,7,10]}
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        param_grid = {'model__n_neighbors':[1,3,4,5,10],\
                      'model__algorithm': ['auto','ball_tree','kd_tree','brute']}
    elif model_name == 'lr':   
        model = LogisticRegression() 
        param_grid = {'model__C': [1,5,10,15,20,30],'model__penalty': ['l2', 'elasticnet', 'none'],\
                      'model__max_iter': [100,150,200,300]}
    elif model_name == 'svm':
        model = SVC()
        param_grid = {'model__C':[1,2,4,5,7,10,15],\
                      'model__kernel':['linear','poly','rbf','sigmoid'],\
                      'model__gamma':['scale','auto']}
    
    pipe = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
    search = GridSearchCV(pipe, param_grid, n_jobs=15)
    search.fit(X_train, y_train)
    print('-----------------------')
    print('Best parameter for {}:'.format(model_name))
    print("CV score=%0.3f:" % search.best_score_)
    print('-----------------------')

    #Remove 'model__' name
    # Code still not clean enough, maybe there is a built-in function which does it???
    if model_name =='svm':
        search.best_params_['C'] = search.best_params_.pop('model__C')
        search.best_params_['kernel'] =search.best_params_.pop('model__kernel')  
        search.best_params_['gamma'] =search.best_params_.pop('model__gamma')
    elif model_name =='knn':
        search.best_params_['n_neighbors'] = search.best_params_.pop('model__n_neighbors')
        search.best_params_['algorithm'] =search.best_params_.pop('model__algorithm')  
    elif model_name =='dt':
        search.best_params_['criterion'] = search.best_params_.pop('model__criterion')
        search.best_params_['min_samples_split'] =search.best_params_.pop('model__min_samples_split') 
        search.best_params_['min_samples_leaf'] =search.best_params_.pop('model__min_samples_leaf') 
        search.best_params_['max_depth'] =search.best_params_.pop('model__max_depth') 
    elif model_name =='lr':
        search.best_params_['C'] = search.best_params_.pop('model__C')
        search.best_params_['penalty'] =search.best_params_.pop('model__penalty')
        search.best_params_['max_iter'] =search.best_params_.pop('model__max_iter')
    elif model_name =='clf':
        search.best_params_['criterion'] = search.best_params_.pop('model__criterion')
        search.best_params_['n_estimators'] = search.best_params_.pop('model__n_estimators')
        search.best_params_['min_samples_split'] =search.best_params_.pop('model__min_samples_split') 
        search.best_params_['min_samples_leaf'] =search.best_params_.pop('model__min_samples_leaf') 
        search.best_params_['max_depth'] =search.best_params_.pop('model__max_depth')

    print(search.best_params_)

    titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model.set_params(**search.best_params_))])
    titanic_pipeline.fit(X_train,y_train)
    print('{}| Cross validation score: {:.3f}'.format(model_name,cross_val_score(titanic_pipeline, X_train, y_train, cv=10).mean()))
    Y_pred[model_name] = titanic_pipeline.predict(X_test)
    
    model_pred = pd.DataFrame({
        "PassengerId": test_df_copy["PassengerId"],
        "Survived": Y_pred[model_name]
    })
    model_pred.to_csv('results/submission_{}.csv'.format(model_name), index=False)
    precision_list.append(compare_with_groundtruth(model_pred))
    print('------------------------------')
    print('{}| Cross validation score: {:.3f}'.format(model_name,cross_val_score(titanic_pipeline, X_train, y_train, cv=10).mean()))



def extract_feature(preprocessor):
    rfc_model = handle_model('clf',preprocessor).fit(X_train, y_train)
    features = pd.DataFrame()
    features['feature'] = test_df.columns
    features['importance'] = rfc_model.steps[1][1].feature_importances_
    '''
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)

    show_plot = True
    if show_plot:
        features.plot(kind='barh', figsize=(25, 25))
        plt.xlabel('Importances',fontsize=18)
        plt.ylabel('feature',fontsize=18)
        plt.savefig('results/feature_importances.pdf')
    rfc_model = SelectFromModel(rfc_model, prefit=True)
    '''

print('Listed Features:')
print('================================')
print(train_df.info())
print(pd.DataFrame(train_df))
print('================================')
#Splitting training data
scaler = StandardScaler()
X_train = train_df.drop(['Survived'], axis=1)
y_train = train_df['Survived'].values

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.2)
#X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = test_df
y_test= pd.read_csv('data/groundtruth.csv')['Survived'].values

print('X_train shape: {}'.format(X_train.shape))
#print('X_val shape: {}'.format(X_val.shape))
print('y_train shape: {}'.format(y_train.shape))
#print('y_val shape: {}'.format(y_val.shape))

Y_pred = pd.DataFrame()
precision_list = []
summary = pd.DataFrame()
model_name = ['svm','knn','dt','lr','clf']
for name in model_name:
    best_para = find_best_para(name,preprocessor)

def ensemble_model():
    Y_pred['majority'] = Y_pred.mode(axis=1)[0].astype(int)

print('==========Combined-result=========')
ensemble_model()
model_pred = pd.DataFrame({
        "PassengerId": test_df_copy["PassengerId"],
        "Survived": Y_pred['majority']
    })
model_pred.to_csv('results/submissions_ensemble.csv', index=False)
precision_list.append(compare_with_groundtruth(model_pred))
print('==================================')

print('==========Summary========')
model_name.append('ensemble')
summary.index = model_name
summary['Precision: Ground Truth'] = precision_list 
print(summary)
print('=========================')