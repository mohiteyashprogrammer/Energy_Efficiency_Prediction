import os
import sys 
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.utils import save_object
from src.utils import model_evaluation

@dataclass
class ModelTraningConfig:
    train_model_file_obj = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_traner_config = ModelTraningConfig()


    def initatied_model_traning(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-2],
                train_array[:, -2:],
                test_array[:,:-2],
                test_array[:, -2:]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic":ElasticNet(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor()

            }
            params = {
                "RandomForestRegressor":{
                    "n_estimators": [100, 200],
                    "max_depth": [2, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "DecisionTreeRegressor":{
                    "max_depth": [2, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "LinearRegression":{
                    
                },
                "Ridge":{
                    "alpha": [0.1, 1, 10],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                },
                "Lasso":{
                    "alpha": [0.1, 1, 10], "max_iter": [1000, 5000, 10000]
                },
                "Elastic":{
                    "alpha": [0.1, 1, 10], "l1_ratio": [0.2, 0.5, 0.8], "max_iter": [1000, 5000, 10000]
                }
            }

            model_report:dict=model_evaluation(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)

                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_object(file_path=self.model_traner_config.train_model_file_obj,
                obj = best_model
                )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)
