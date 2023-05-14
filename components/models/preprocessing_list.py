import numpy as np
import pandas as pd
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
class predictions_1 :
        def __init__(self, df_list):
            self.df_list = df_list

        # def covert_to_datafrme(self,df_list):
        #        df = pd.DataFrame(self.df_list, columns=["Make","Model","Year","Engine Fuel Type","Engine HP","Engine Cylinders","Transmission Type","Driven_Wheels","Number of Doors","Market Category","Vehicle Size","Vehicle Style","highway MPG","city mpg","Popularity","MSRP"])
        #        return df
               
        def multihot_encode(self,df, column):
                df = df.copy()
                
                df[column] = df[column].apply(lambda x: x.split(','))
                
                all_categories = np.unique(df[column].sum())
                
                for category in all_categories:
                    df[column + '_' + category] = df.apply(lambda x: 1 if category in x[column] else 0, axis=1)
                
                df = df.drop(column, axis=1)
                
                return df
        def onehot_encode(self,df, column):
                df = df.copy()
                dummies = pd.get_dummies(df[column], prefix=column)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(column, axis=1)
                return df
        def preprocess_inputs(self,df):
                df = df.copy()
                
                # Fill multi-hot column missing values
                df['Market Category'] = df['Market Category'].fillna("Missing")
                
                # Multi-hot encoding
                df = self.multihot_encode(df, column='Market Category')
                
                # One-hot encoding
                for column in df.select_dtypes('object').columns:
                    df = self.onehot_encode(df, column=column)
                
                # Fill remaining missing values
                df['Engine HP'] = df['Engine HP'].fillna(df['Engine HP'].mean())
                for column in ['Engine Cylinders', 'Number of Doors']:
                    df[column] = df[column].fillna(df[column].mode()[0])
                
                # Split df into X and y
                df.to_csv('data_lean.csv')
                y = df['MSRP']
                X = df.drop('MSRP', axis=1)
                return y,X
        def split(self,df):
                y,X = self.preprocess_inputs(df)
                ll=X[:1]
                kk =ll.to_json(orient='records')

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
                
                
                # Scale X
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
                
                return X_train, X_test, y_train, y_test
        def final_model (self):
                df = self.df_list
                X_train, X_test, y_train, y_test = self.split(df)
                model=RandomForestRegressor()  
                print("X_train") 
                print(X_train)
                print("X_train_type",type(X_train)) 
                print(X_train[:0])
                model.fit(X_train, y_train)  

                return model 
        def prrr(self,data):
                model = self.final_model()
                pred= model.predict(data)
                return pred
        
        
                        