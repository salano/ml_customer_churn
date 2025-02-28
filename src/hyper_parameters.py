import numpy as np
from scipy.stats import randint, uniform

params={
            "logistic Regression": {
                'penalty' : ['l1','l2'], 
                'C'       : np.logspace(-3,3,7),
                 'penalty' : ['l1','l2'],
                'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
            },
            "Support Vector Machines":{
                'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']
            },
            "k-Nearest Neighbors":{
                'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']
            },
            "Naive Bayes":{
                'var_smoothing': np.logspace(0,-9, num=100)
            },
            "Decision Trees":{
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Random Forest":{
                'n_estimators': [25, 50, 100, 150],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [3, 6, 9],
                'max_leaf_nodes': [3, 6, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_samples': [0.5, 0.75, 1.0]
            },
            "XGBoost":{
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.01, 0.001],
                'subsample': [0.5, 0.7, 1]

            },
            "LightGBM Classifier":{
                'objective': ['multiclass', 'binary'],
                #'num_class': [2],  
                #'metric': 'multi_logloss',
                #'boosting_type': 'gbdt',
                'num_leaves': [5, 20, 31],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 150]
            },
            "CatBoost Classifier":{
                'learning_rate': [0.03, 0.1],
                'depth': [4, 6, 10],
                'l2_leaf_reg': [1, 3, 5,]
            },
            "Neural Networks":{
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['adam', 'sgd'],
                'max_iter': [50, 100, 150]
            }
                
        }


'''
from sklearn.model_selection import RepeatedStratifiedKFold

cv_method = RepeatedStratifiedKFold(n_splits=5, 
                                    n_repeats=3, 
                                    random_state=999)

gs_NB = GridSearchCV(estimator=model, 
                     param_grid=params_NB, 
                     cv=cv_method,
                     verbose=1, 
                     scoring='accuracy')
'''