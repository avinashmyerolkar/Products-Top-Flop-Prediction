#!/usr/bin/env python
# coding: utf-8

# # Building ANN_sequential model on ('historic_data')training data and predicting inputs given in 'prediction_inpu.csv' file

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('train_df.csv')  # will be using file which encoded in EDA file for modelling purpose
df.head(1)


# # Modelling data on keras_classifier

# In[4]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')


# In[5]:


class ANNClassifier:
    def __init__(self):                   #initializing variables
        self.pipeline = None
        self.label_encoder1 = None
        self.label_encoder2 = None
        self.label_encoder3 = None
        self.scaler = None
        self.x_train = None
        self.x_test = None
        self.y_encoded_train = None
        self.y_encoded_test = None

    def load(self, file_path):                    # Loading data stage
        self.data = pd.read_csv(file_path)

    def preprocess(self):  # preprocessing stage

        x = self.data.drop(['success_indicator', 'item_no'], axis=1)  # separating dependent and independent variables
        y = self.data['success_indicator']

        # here we will encode all text categorical data to numerical categorical data
        self.label_encoder1 = LabelEncoder()
        x['category_encoded'] = self.label_encoder1.fit_transform(x['category'])
        x.drop('category', axis=1, inplace=True)

        self.label_encoder2 = LabelEncoder()
        x['main_promotion_encoded'] = self.label_encoder2.fit_transform(x['main_promotion'])
        x.drop('main_promotion', axis=1, inplace=True)

        self.label_encoder3 = LabelEncoder()
        x['color_encoded'] = self.label_encoder3.fit_transform(x['color'])
        x.drop('color', axis=1, inplace=True)

        # here we will convert stars rating features into bins where rating <=3 will be 0 and rating>3 will be 1
        x['stars'] = np.where(x['stars'] <= 3, 0, x['stars'])
        x['stars'] = np.where(x['stars'] > 3, 1, x['stars'])

        # encoding category where 'flop'= 0 and 'top'=1
        label_encoder4 = LabelEncoder()
        y_encoded = label_encoder4.fit_transform(y)
        y_encoded = np.where(y_encoded == label_encoder4.classes_.tolist().index('flop'), 0, y_encoded)
        y_encoded = np.where(y_encoded == label_encoder4.classes_.tolist().index('top'), 1, y_encoded)

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x)

        # here we will split the data into training and testing purpose
        self.x_train, self.x_test, self.y_encoded_train, self.y_encoded_test = train_test_split(x_train, y_encoded,
                                                                                                test_size=0.2,
                                                                                                random_state=77)

    def create_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self):
        # Creating KerasClassifier by using above model
        keras_model = KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=32, verbose=0)

        # Creating sklearn pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Preprocessing step, if needed
            ('classifier', keras_model)  # Keras model as the classifier
        ])

        # here we will fit the pipeline on the training data
        self.pipeline.fit(self.x_train, self.y_encoded_train)

    def predict(self):
        # we will use pipeline to make predictions
        return self.pipeline.predict(self.x_test)
    
    def evaluate(self):
        
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_encoded_test, y_pred)
       
        precision = precision_score(self.y_encoded_test,y_pred)
        recall = recall_score(self.y_encoded_test, y_pred)
        f1 = f1_score(self.y_encoded_test, y_pred)
        
        print("Accuracy with ann_classifier is:", accuracy)
        print("precision with  ann_classifier model is :", precision)  # Model evaluation stage
        print("recall with ann_classifier model is:", recall)
        print("f1 Score with ann_classifier model is :", f1)
     
   # function to load unlabelled file i.e, 'prediction_input.csv'
    def load_test_file(self, file_path):                                      #loading stage for testing file
        self.input_data = pd.read_csv(file_path)
        return self.input_data
        
        # this function will process input data such as removing the unwanted features and encoding categorical features
    def test_data_preprocessor(self):
       
        self.input_data_processed = self.input_data.drop(['item_no'], axis=1)           #preprocessing stage for 
        
        self.input_data_processed['category_encoded'] = pipeline.label_encoder1.transform(self.input_data_processed['category'])
        self.input_data_processed.drop('category', axis=1, inplace=True)
        
        self.input_data_processed['main_promotion_encoded'] = pipeline.label_encoder2.transform(self.input_data_processed['main_promotion'])
        self.input_data_processed.drop('main_promotion', axis=1, inplace=True)
        
        self.input_data_processed['color_encoded'] = pipeline.label_encoder3.transform(self.input_data_processed['color'])
        self.input_data_processed.drop('color', axis=1, inplace=True)
        
        # here we will convert star rating into bins and designate star <= 3 into 0 and star rating> 3 =1
        self.input_data_processed['stars'] = np.where(self.input_data_processed['stars'] <= 3, 0, self.input_data_processed['stars'])
        self.input_data_processed['stars'] = np.where(self.input_data_processed['stars'] > 3, 1, self.input_data_processed['stars'])
        
        self.input_data_processed = pipeline.scaler.transform(self.input_data_processed)
        
        return self.input_data_processed
        
        
        # this function will predict dependent variable based on independent variable present in prediction_input file
            # and give a an array of 2000 row in form of 1 , 0 where 1 = 'TOP' AND 0 = 'FLOP'
    def predict_for_test_data(self):
        output = self.pipeline.predict(self.input_data_processed)
        return output
                           
    
        


# In[6]:


pipeline = ANNClassifier()
pipeline.load('historic.csv')
pipeline.preprocess()
pipeline.create_model()


# In[7]:


pipeline.fit()
pipeline.predict()
pipeline.evaluate()


# # PREDICTION ON 'prediction_input.csv' using ANN sequential model

# In[8]:


pipeline.load_test_file('prediction_input.csv')
pipeline.test_data_preprocessor()


# In[9]:


print('required array of prediction of classes 2 classes on 2000 rows : ',pipeline.predict_for_test_data())

