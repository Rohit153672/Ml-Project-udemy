import os
import sys
import numpy as np
import dill
import pandas as pd
from sklearn.metrics import r2_score
from src.logger import logging


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name} -> Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

            # Save test R2 score to report
            report[model_name] = test_score

        return report

        
    
    except Exception as e:
        raise CustomException(e,sys)

    

  