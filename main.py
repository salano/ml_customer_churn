from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import numpy as np
import pandas as pd


from src.pipelines.predict_pipeline import CustomData, PredictPipeline

if __name__ == "__main__":
    #Uncomment the below to retrain the model
    
    data_file_path = r'notebook\data\Churn_Modelling.csv'
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion(data_file_path)
    tgt_column_name = 'Exited'
    r_columns = ['RowNumber', 'Surname', 'CustomerId','Exited']

    #dynamic categorization
    train_df = pd.read_csv(train_data)
    numeric_lst=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_columns = list(train_df.select_dtypes(include=numeric_lst).columns)
    cat_columns = train_df.select_dtypes(include="object").columns.tolist()

    #remove unwanted columns
    for col in r_columns:
        if col in num_columns:
            num_columns.remove(col)
        else:
            cat_columns.remove(col)

    data_transformation = DataTransformation()
    data_transformation.get_data_transformation_object(numerical_columns=num_columns, categorical_columns=cat_columns)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data, 
        numerical_columns=num_columns, categorical_columns=cat_columns,
        target_column_name=tgt_column_name, removed_columns=r_columns
        )

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    # Make a prediction on unseen data
    data = CustomData(
            CreditScore = 675,
            Geography = 'France',
            Gender = 'Female',
            Age = 45,
            Tenure = 7,
            Balance = 10000,
            NumOfProducts = 3,
            HasCrCard = 0,
            IsActiveMember = 1,
            EstimatedSalary = 125850.75

    )
  
    predict_df = data.get_data_as_dataframe()
    print(predict_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(predict_df)
    print("Predicted Class: {:.4f}".format(results[0]))