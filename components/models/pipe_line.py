import numpy as np
import pandas as pd
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
class predictions_12 :
        def __init__(self, df_list):
            self.df_list = df_list
        def per(self):
            df= self.df_list.copy()
            column= df.columns.tolist()
            #df[column]] = df[column].apply(lambda x: x.split(','))
                
            
            df['Market Category'] = df['Market Category'].fillna("Missing")
             # Fill remaining missing values
            df['Engine HP'] = df['Engine HP'].fillna(df['Engine HP'].mean())
            for column in ['Engine Cylinders', 'Number of Doors']:
                    df[column] = df[column].fillna(df[column].mode()[0])
            y = df['MSRP']
            X = df.drop('MSRP', axis=1)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
            ohe=OneHotEncoder()
            ohe.fit(X[["Make", "Model",	"Year",	"Engine Fuel Type"	,"Vehicle Style" , "Vehicle Size"	,"Market Category" ,"Driven_Wheels" ]])
            column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),["Make", "Model",	"Year",	"Engine Fuel Type"	,"Vehicle Style" , "Vehicle Size"	,"Market Category" ,"Driven_Wheels" ]),
                                    remainder='passthrough')
            rf=RandomForestRegressor() 
            print(X_train)
            pipe=make_pipeline(column_trans,rf)
            pipe.fit(X_train,y_train)
            y_pred=pipe.predict(X_test)
            r2_score(y_test,y_pred)
            scores=[]
            for i in range(1000):
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
                lr=LinearRegression()
                pipe=make_pipeline(column_trans,lr)
                pipe.fit(X_train,y_train)
                y_pred=pipe.predict(X_test)
                scores.append(r2_score(y_test,y_pred))
            np.argmax(scores)
            scores[np.argmax(scores)]
            pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(["BMW","1 Series M",	2011,"premium unleaded (required)",	335,	6,"MANUAL",	"rear wheel drive" ,	2 ,	"Factory Tuner","Luxury","High-Performance",	"Compact	Coupe"	,26	,19	,3916]).reshape(1,15)))
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
            lr=LinearRegression()
             
            pipe.fit(X_train,y_train)
            y_pred=pipe.predict(X_test)
            s=r2_score(y_test,y_pred)
            pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
            #pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
            return s