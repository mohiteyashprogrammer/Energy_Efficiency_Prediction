import os
import sys 
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
##Pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")

    #Create Data TRansdormation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_trainsformation_object(self):
        try:
            logging.info("Data Transformation Started")

            numerical_features = ['relative_compactness', 'wall_area', 'overall_height', 'orientation',
            'glazing_area', 'glazing_area_distribution']

            logging.info("Pipline Started")

            ## Numerical Pipline
            num_pipline = Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )

            # create Preprocessor Object
            preprocessor = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Error Occured In Data TRansformation Class")
            raise CustomException(e, sys)


    def inited_data_transformation(self,train_path,test_path):
        try:
            ## Read Train And Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning and TEst Data Complited")
            logging.info(f"Train DataFrame Head : \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n{test_data.head().to_string()}")

            logging.info("Obtaning Prrprocessor Object")

            preprocessor_obj = self.get_data_trainsformation_object()

            target_columns_name = ["heating_load","cooling_load"]
            drop_column = [target_columns_name]
            

            ## Split dependent nd independent features
            input_features_train_data = train_data.drop(labels=['heating_load', 'cooling_load'],axis=1)
            target_feature_train_data = train_data[target_columns_name]
            

            ## Split dependent nd independent features
            input_features_test_data = test_data.drop(['heating_load', 'cooling_load'],axis=1)
            target_feature_test_data = test_data[target_columns_name]
    

            ## Apply Transformton Object on train and test data
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)


            logging.info("Apply Preprocessor Object on train and test Data")

            ## Convert in To Array To BEcome Faster
            train_array = np.c_[input_features_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            

            # CAlling Saving Object and Asve Preprocessing Object
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
            )

            logging.info("Preproccsing Pickel File Save")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception  as e:
            logging.info("Error Occured In Data Transformation Class")
            raise CustomException(e, sys)

