import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from feature_engine.creation import MathematicalCombination
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define the categorical variables
            CATEGORICAL_VARS = ['Month',
                'OperatingSystems',
                'Browser',
                'Region',
                'TrafficType',
                'VisitorType',
                'Weekend']
 # update these with your actual categorical column names

            # set up the pipeline
            purchase_pipe = ([
                          ('categorical_encoder', OneHotEncoder(variables=CATEGORICAL_VARS)),
                          ('feature_creation', MathematicalCombination(
                              variables_to_combine=['Administrative_Duration',
                                                    'Informational_Duration',
                                                    'ProductRelated_Duration'
                                                    ],
                              math_operations=[
                                               'sum'],      
                              new_variables_names=['duration']        
                          )),
                          
       #                   ('drop feature', DropFeatures(
    #features_to_drop=['Administrative_Duration',
       #                                             'Informational_Duration',
       #                                             'ProductRelated_Duration'])),
                          ('yeojohnson', PowerTransformer(method='yeo-johnson')),
                          
                          ('scaler', MinMaxScaler()),
                          ('smote', SMOTE(sampling_strategy=0.3)),
                          ('under', RandomUnderSampler(sampling_strategy=0.5)),
                          ('Catboost', CatBoostClassifier(iterations=1000,                          
                           devices='0:1',
                           verbose=0)),
                          ])

            logging.info(f"Categorical columns: {CATEGORICAL_VARS}")

            return purchase_pipe

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            purchase_pipe=self.get_data_transformer_object()

            target_column_name="target"  # update this with your actual target column name

            input_feature_train_df=train_df.drop(columns=['Revenue'],axis=1)
            target_feature_train_df=train_df['Revenue']

            input_feature_test_df=test_df.drop(columns=['Revenue'],axis=1)
            target_feature_test_df=test_df['Revenue']

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=purchase_pipe.fit_transform(input_feature_train_df)
            input_feature_test_arr=purchase_pipe.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=purchase_pipe
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
