import sys
import pandas as pd

from src.exception import CustomException
from src.exception import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = 'artifacts\model.pk1'
            preprocessor_path = 'artifacts\preprocessor.pk1'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            predicts = model.predict(data_scaled)

            return predicts
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self,
                 CreditScore: int,
                 Geography: str,
                 Gender: str,
                 Age: int,
                 Tenure: int,
                 Balance: float,
                 NumOfProducts: int,
                 HasCrCard: int,
                 IsActiveMember: int,
                 EstimatedSalary: float
                 
                 ):
        
        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'CreditScore' : [self.CreditScore],
                'Geography' : [self.Geography],
                'Gender' : [self.Gender],
                'Age' : [self.Age],
                'Tenure' : [self.Tenure],
                'Balance' : [self.Balance],
                'NumOfProducts' : [self.NumOfProducts],
                'HasCrCard' : [self.HasCrCard],
                'IsActiveMember' : [self.IsActiveMember],
                'EstimatedSalary' : [self.EstimatedSalary]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)