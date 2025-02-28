import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.hyper_parameters import params


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pk1')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('splitting train and test datasets')
            X_train, Y_train, X_test, Y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
        
            models = {
                #'logistic Regression': LogisticRegression(),
                #'Support Vector Machines': svm.SVC(kernel='linear'),
                #'k-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
                #'Naive Bayes': GaussianNB(),
                #'Decision Trees': DecisionTreeClassifier(),
                #'Random Forest': RandomForestClassifier(),
                #'XGBoost': XGBClassifier(),
                #'LightGBM Classifier': lgb.LGBMClassifier(num_class=2, boosting_type='gbdt'),
                'CatBoost Classifier': CatBoostClassifier(verbose=False),
                #'Neural Networks' : MLPClassifier()

            }

            model_report: dict = evaluate_model(x_train=X_train,
                                                y_train=Y_train, x_test=X_test,
                                                y_test=Y_test, models=models,
                                                param=params)

            # Get best model score
            best_model_score = max(sorted(model_report.values()))

            #Get Best Model Name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No suitable model found')
            else:
                logging.info("Best model found on training dataset is [{}]".format(best_model))

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            pred_df = pd.DataFrame({'Actual Value': 
                                  Y_test, 'Predicted Value': predicted,
                                  'Difference': Y_test-predicted})
            print(pred_df)

            acc_score = accuracy_score(Y_test, predicted)

            return acc_score
            
        except Exception as e:
            raise CustomException(e, sys)

