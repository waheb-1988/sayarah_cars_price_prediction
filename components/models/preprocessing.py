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

class predictions :
        def __init__(self, path, path_output, file_name):
            self.path = path
            self.file_name= file_name
            self.path_output= path_output
            

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
                y = df['MSRP']
                X = df.drop('MSRP', axis=1)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
                
                # Scale X
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
                
                return X_train, X_test, y_train, y_test
        def final_model (self,df):
                X_train, X_test, y_train, y_test = self.preprocess_inputs(df)
                print(X_train)
                n_components = 100

                pca = PCA(n_components=n_components)
                pca.fit(X_train)
                
                X_train_reduced = pd.DataFrame(pca.transform(X_train), index=X_train.index, columns=["PC" + str(i) for i in range(1, n_components + 1)])
                X_test_reduced = pd.DataFrame(pca.transform(X_test), index=X_test.index, columns=["PC" + str(i) for i in range(1, n_components + 1)])
                models = {
                        # "                     Linear Regression": LinearRegression(),
                        # " Linear Regression (L2 Regularization)": Ridge(),
                        # " Linear Regression (L1 Regularization)": Lasso(),
                        # "                   K-Nearest Neighbors": KNeighborsRegressor(),
                        # "                        Neural Network": MLPRegressor(),
                        # "Support Vector Machine (Linear Kernel)": LinearSVR(),
                        # "   Support Vector Machine (RBF Kernel)": SVR(),
                        # "                         Decision Tree": DecisionTreeRegressor(),
                        "                         Random Forest": RandomForestRegressor()
                         # "                     Gradient Boosting": GradientBoostingRegressor()
                }

                for name, model in models.items():
                        model.fit(X_train_reduced, y_train)
                        print(name + " trained.")
                for name, model in models.items():
                        print(name + " R^2 Score: {:.5f}".format(model.score(X_test_reduced, y_test)))
                return model 