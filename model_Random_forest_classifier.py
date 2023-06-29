#!/usr/bin/env python
# coding: utf-8

# # Building Random forest model on ('historic_data')training data and predicting inputs given in 'prediction_inpu.csv' file

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('historic.csv')


# In[7]:


df.head(1)


# In[4]:


# FOR building this class we will take into consideration insights we drawn in our EDA report file 

class Random_forest_pipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder1 = None
        self.label_encoder2 = None
        self.label_encoder3 = None
        
    
 
    def load(self, file_path):                             # Loading data stage
        self.data = pd.read_csv(file_path)

    def preprocess(self):                                  # preprocessing stage
    
        x = self.data.drop(['success_indicator', 'item_no'], axis=1)  # seperating dependent and independent varaiables
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
        
        #here we will convert stars rating features into bins where rating <=3 will be 0 and rating>3 will be 1
        x['stars'] = np.where(x['stars'] <= 3, 0, x['stars']) 
        x['stars'] = np.where(x['stars'] > 3, 1, x['stars'])
        
        
        # encoding category where 'flop'= 0 and 'top'=1
        label_encoder4 = LabelEncoder()
        y_encoded = label_encoder4.fit_transform(y)
        y_encoded = np.where(y_encoded == label_encoder4.classes_.tolist().index('flop'), 0, y_encoded)
        y_encoded = np.where(y_encoded == label_encoder4.classes_.tolist().index('top'), 1, y_encoded)

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x)
        
        #here we will split the data into training and testing purpose
        self.x_train, self.x_test, self.y_encoded_train, self.y_encoded_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
        

        #function for training the model with historic data
    def train(self):
        self.rf_clf = RandomForestClassifier()
        self.rf_clf.fit(self.x_train, self.y_encoded_train)                # training data stage
        
        #function for testing the model with historic data
    def test(self):                                                        # testing data stage
        y_pred = self.rf_clf.predict(self.x_test)
        

       # evaluating the model performance with accuaccuracy_score , precision, recall and f1_score
        accuracy = accuracy_score(self.y_encoded_test, y_pred)
        precision = precision_score(self.y_encoded_test,y_pred)
        recall = recall_score(self.y_encoded_test, y_pred)
        f1 = f1_score(self.y_encoded_test, y_pred)
        
        print("Accuracy with random_forest_classifier is:", accuracy)
        print("precision with Decision tree classifier model is :", precision)  # Model evaluation stage
        print("recall with Decision tree classifier model is:", recall)
        print("f1 Score with Decision tree classifier model is :", f1)
        
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
        output = pipeline.rf_clf.predict(self.input_data_processed)
        return output
        
                           
        
        
       
    
        
        


# In[5]:


pipeline = Random_forest_pipeline()
pipeline.load('historic.csv')
pipeline.preprocess()
pipeline.train()
pipeline.test()


# # PREDICTION ON 'prediction_input.csv'

# In[6]:


pipeline.load_test_file('prediction_input.csv')
pipeline.test_data_preprocessor()
print('prediction on 2000 rows in prediction_input data : ',pipeline.predict_for_test_data().shape) 
print('required array of prediction of classes 2 classes on 2000 rows : ',pipeline.predict_for_test_data())


# In[ ]:




