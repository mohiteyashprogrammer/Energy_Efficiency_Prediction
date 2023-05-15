import os
import sys 
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            ## Load pickel File
            ## This Code Work in any system

            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            # Load object
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occure in Prediction Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
            relative_compactness:float,
            wall_area:float,
            overall_height:float,
            orientation:int,
            glazing_area:float,
            glazing_area_distribution:int,
            ):

        self.relative_compactness = relative_compactness
        self.wall_area = wall_area
        self.overall_height = overall_height
        self.orientation = orientation
        self.glazing_area = glazing_area
        self.glazing_area_distribution = glazing_area_distribution


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "relative_compactness":[self.relative_compactness],
                "wall_area":[self.wall_area],
                "overall_height":[self.overall_height],
                "orientation":[self.orientation],
                "glazing_area":[self.glazing_area],
                "glazing_area_distribution":[self.glazing_area_distribution]
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Data Frame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Predict Pipline")
            raise CustomException(e, sys)

    
          
