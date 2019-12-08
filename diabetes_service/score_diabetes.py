import json
import numpy as np
import os
import joblib
from azureml.core.model import Model
# import azureml.train.automl # Required for AutoML models with pre-processing

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('diabetes')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data - the features of patients to be classified.
    data = json.loads(raw_data)['data']
    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)