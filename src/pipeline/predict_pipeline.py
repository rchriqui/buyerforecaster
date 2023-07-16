import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 Month: str,
                 VisitorType: str,
                 Administrative: float,
                 Administrative_Duration: float,
                 Informational: float,
                 Informational_Duration: float,
                 ProductRelated: float,
                 ProductRelated_Duration: float,
                 BounceRates: float,
                 ExitRates: float,
                 PageValues: float,
                 SpecialDay: float,
                 OperatingSystems: int,
                 Browser: int,
                 Region: int,
                 TrafficType: int):
        
        self.Month = Month
        self.VisitorType = VisitorType
        self.Administrative = Administrative
        self.Administrative_Duration = Administrative_Duration
        self.Informational = Informational
        self.Informational_Duration = Informational_Duration
        self.ProductRelated = ProductRelated
        self.ProductRelated_Duration = ProductRelated_Duration
        self.BounceRates = BounceRates
        self.ExitRates = ExitRates
        self.PageValues = PageValues
        self.SpecialDay = SpecialDay
        self.OperatingSystems = OperatingSystems
        self.Browser = Browser
        self.Region = Region
        self.TrafficType = TrafficType

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Month': [self.Month],
                'VisitorType': [self.VisitorType],
                'Administrative': [self.Administrative],
                'Administrative_Duration': [self.Administrative_Duration],
                'Informational': [self.Informational],
                'Informational_Duration': [self.Informational_Duration],
                'ProductRelated': [self.ProductRelated],
                'ProductRelated_Duration': [self.ProductRelated_Duration],
                'BounceRates': [self.BounceRates],
                'ExitRates': [self.ExitRates],
                'PageValues': [self.PageValues],
                'SpecialDay': [self.SpecialDay],
                'OperatingSystems': [self.OperatingSystems],
                'Browser': [self.Browser],
                'Region': [self.Region],
                'TrafficType': [self.TrafficType]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)