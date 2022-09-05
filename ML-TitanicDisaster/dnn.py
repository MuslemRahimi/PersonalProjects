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
from keras.layers import Dropout,Dense,BatchNormalization
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


# ratio of test/train
#print(len(test_df)/(len(train_df)+len(test_df)))


#Add feature: Family Size
for dataset in df:
    dataset['Fam_size'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Fam_type'] = pd.cut(dataset.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big']).astype(object)

for dataset in df:
    '''
    dataset['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
    dataset['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer','Rev'], 'Mr', inplace=True)
    '''
    # creation of the Title feature
    dataset['Title'] = 'man'
    dataset.loc[dataset.Sex == 'female', 'Title'] = 'woman'
    dataset.loc[dataset['Name'].str.contains('Master'), 'Title'] = 'boy'
'''
print(train_df['Title'].value_counts())
plt.title('Survival rate by Title')
g = sns.barplot(x='Title', y='Survived', data=train_df).set_ylabel('Survival rate')
plt.show()
'''

for dataset in df:
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
    dataset['Age'] = np.log(dataset['Age']+1)
    #dataset['Age_type'] = pd.cut(dataset['Age'], [16,32,64], labels=[0, 1]).astype(float)


for dataset in df:
    # Extract the first two letters
    dataset['Ticket_lett'] = dataset.Ticket.apply(lambda x: x[:2])
    # Calculate ticket length
    dataset['Ticket_len'] = dataset.Ticket.apply(lambda x: len(x))

#Add feature: Deck
for dataset in df:
    dataset['Deck'] = dataset['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

for dataset in df:
    idx= dataset[dataset['Deck'] == 'T'].index
    dataset.loc[idx, 'Deck'] = 'A'

for dataset in df:
    dataset['Deck'] = dataset['Deck'].replace(['A','B','C'], 0)
    dataset['Deck'] = dataset['Deck'].replace(['D','E',], 1)
    dataset['Deck'] = dataset['Deck'].replace(['F','G'], 2)
    dataset['Deck'] = dataset['Deck'].replace(['M'], 3)
    dataset['Deck'] = dataset['Deck'].astype(int)

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

# Dropping irrelevant column features
drop_columns = ['PassengerId','Name',\
               'Cabin', 'Embarked','Ticket_lett',\
               'Ticket','Ticket_len','Sex']

'''
The categorical features (Pclass, Sex, Deck, Embarked, Title) are converted 
to one-hot encoded features with OneHotEncoder. Age and Fare features are not 
converted because they are ordinal unlike the previous ones.
'''


for dataset in df:
    dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
    dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1
    dataset['Sex'] = dataset['Sex'].astype(int)

'''
g = sns.barplot(x=train_df['Fam_type'], y=train_df['Survived']).set_ylabel('Survival rate')
plt.show()
'''


'''
df_cat = train_df[['Survived', 'Pclass', 'Sex','Age_type']]
## Correlation Matrix
plt.subplots(figsize=(10,7))
sns.heatmap(df_cat.corr(), cmap='Blues', annot=True, linewidths=2, annot_kws={"fontsize":15})
plt.show()
'''


categorical_cols = ['Pclass','Title',\
                   'Deck','Fam_size','Fam_type','Ticket_len']

encoded_features = []
for dataset in df:
    for feature in categorical_cols:
        encoded_feat = OneHotEncoder().fit_transform(dataset[feature].values.reshape(-1, 1)).toarray()
        n = dataset[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_dataset = pd.DataFrame(encoded_feat, columns=cols)
        encoded_dataset.index = dataset.index
        encoded_features.append(encoded_dataset)

train_df = pd.concat([train_df, *encoded_features[:len(categorical_cols)]], axis=1)
test_df = pd.concat([test_df, *encoded_features[len(categorical_cols):]], axis=1)


train_df.drop(drop_columns+categorical_cols, inplace=True, axis=1)
test_df.drop(drop_columns+categorical_cols, inplace=True, axis=1)
df = [train_df, test_df]


print('Listed Features:')
print('================================')
print(train_df.info())
print(test_df.info())
print('================================')
#Splitting training data
X_train = train_df.drop(['Survived'], axis=1)
y_train = train_df['Survived'].values

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.2)
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = test_df
X_test = pd.DataFrame(scaler.fit_transform(X_test),columns=X_test.columns)
y_test= pd.read_csv('data/groundtruth.csv')['Survived'].values

print('X_train shape: {}'.format(X_train.shape))
print('X_test shape: {}'.format(X_test.shape))
print('y_train shape: {}'.format(y_train.shape))
print('y_test shape: {}'.format(y_test.shape))

Y_pred = pd.DataFrame()


#---------------Feature selection---------------# 
clf = RandomForestClassifier()
param_grid = {'criterion':['entropy','gini','log_loss'],\
              'n_estimators': [50,100,150,200],\
              'min_samples_split':[1,5,7,9],\
              'min_samples_leaf': [1,5,7,9],\
              'max_depth': [1,5,7,9]}

search = GridSearchCV(clf, param_grid, n_jobs=-1)
search.fit(X_train, y_train)
print('-----------------------')
print("CV score=%0.3f:" % search.best_score_)
print('Best parameters: ', search.best_params_)
print('-----------------------')
clf = clf.set_params(**search.best_params_).fit(X_train, y_train)
clf = SelectFromModel(clf, prefit=True)

X_train = clf.transform(X_train)
X_test = clf.transform(X_test)


print('---------------------------------')
print('Reduced feature set X_train shape: ',X_train.shape)
print('Reduced feature set X_test shape: ',X_test.shape)
print('---------------------------------')

#----------------------------------------------#


class DNN():
    def __init__(self):
        self.input_feature = X_train.shape[1]
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9)
        #self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-4)
        self.neuralnetwork = self.build_model()
        self.neuralnetwork.compile(optimizer=self.optimizer,
                                   loss="mse",
                                   metrics=["accuracy"])
    
    def build_model(self):
        nn_model = tf.keras.models.Sequential()
        nn_model.add(Dense(512, activation='leaky_relu', kernel_initializer='he_normal', input_shape=(self.input_feature,)))
        nn_model.add(Dropout(0.5))
        nn_model.add(BatchNormalization())
        nn_model.add(Dense(256, activation='leaky_relu', kernel_initializer='he_normal'))
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(128, activation='leaky_relu', kernel_initializer='he_normal'))
        nn_model.add(Dropout(0.5))
        nn_model.add(Dense(10, activation='leaky_relu', kernel_initializer='he_normal'))
        nn_model.add(Dense(1, activation='sigmoid'))
        return nn_model

    def train_model(self,epochs,batch_size=1024,save_interval=50):
        '''
        
        self.neuralnetwork.trainable = True
        for epoch in range(epochs):
            # -----------------------
            #      Train Model
            # -----------------------
            train_loss = self.neuralnetwork.train_on_batch(X_train,y_train)
            val_loss = self.neuralnetwork.test_on_batch(X_val,y_val)
            #train_loss_list.append(train_loss[0])
            #train_accuracy_list.append(train_loss[1])

            #val_loss_list.append(val_loss[0])
            #val_accuracy_list.append(val_loss[1])

            if epoch % save_interval == 0:
                print ("Epoch: %d|Training: [train_loss: %f, train_acc.: %.2f%%]|Validation: [val_loss: %f, val_acc.: %.2f%%]" % \
                    (epoch, train_loss[0], 100*train_loss[1],val_loss[0], 100*val_loss[1]))
                
        '''
        self.neuralnetwork.fit(
                            X_train,    
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            use_multiprocessing=True
                        )

    def predict_model(self):
        #self.neuralnetwork.evaluate(X_test,y_test)

        pred = self.neuralnetwork.predict(X_test)
        pred_list=[]
        for i in range(len(pred)):
            if pred[i] >= 0.5:
                num = 1
            else:
                num = 0
            pred_list.append(num)
        return pred_list
 

def submit_model():
    testmodel = DNN()
    testmodel.train_model(epochs=2000,batch_size=100,save_interval=500)
    Y_pred['nn'] = testmodel.predict_model()

    nn_pred = pd.DataFrame({
        "PassengerId": test_df_copy["PassengerId"],
        "Survived": Y_pred['nn']
        })
    nn_pred.to_csv('results/submissions_nn.csv', index=False)
    print('Neural-Network ground_truth:')
    compare_with_groundtruth(nn_pred)
    #===================================#
    
print("Submission process: Started!")
submit_model()
print("Submission process: Finished!")
