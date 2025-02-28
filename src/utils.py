import os
import sys 
import dill

import numpy as np 
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

targets = ['Not churned', 'Churned']


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
  
    
def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    try:
        report: dict = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            logging.info("Train model [{}]".format(list(models.keys())[i]))
            print(list(models.keys())[i],'\n')
            #Hyperparameter
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            #end Hyperparameter
            model.fit(x_train, y_train)

            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)


            train_model_acc_score = accuracy_score(y_train, y_train_predict)
            test_model_acc_score = accuracy_score(y_test, y_test_predict)

            train_model_auc_score = roc_auc_score(y_train, y_train_predict)
            test_model_auc_score = roc_auc_score(y_test, y_test_predict)

            cm = confusion_matrix(y_test, y_test_predict)
            cr = classification_report(y_test, y_test_predict, target_names=targets)

            print('Model performance for Training set')
            print("- Accuracy Score: {:.4f}".format(train_model_acc_score))
            print("- ROC AUC : {:.4f}".format(train_model_auc_score))

            print('----------------------------------')
            
            print('Model performance for Test set')
            print("- Accuracy Score: {:.4f}".format(test_model_acc_score))
            print("- ROC AUC: {:.4f}".format(test_model_auc_score))

            print('----------------------------------\n')

            print("Confusion Matrix:")
            print(cm)

            print('----------------------------------\n')

            print("Classification Report:\n")
            print(cr)

            report[list(models.keys())[i]] = test_model_acc_score

        print(pd.DataFrame(report.items(), columns=['Model Name', 'Accuracy Score']).sort_values(by=["Accuracy Score"],ascending=False))

        return report
            
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
