#!/usr/bin/python3
import os
import cgi
import numpy as np
import json
from tensorflow.keras.models import load_model

# Set the appropriate environment for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU if GPU is having issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Directory where models are saved
MODEL_DIR = 'models'

def handle_file_upload():
    form = cgi.FieldStorage()
    dataset = form.getvalue("dataset")
    task = form.getvalue("task")

    # Get the uploaded files
    files = [form.getvalue(f"file{i}") for i in range(len(form) - 2)]  # Ignore dataset and task fields

    # Perform the prediction based on the selected task
    try:
        predictions = perform_prediction(dataset, task, files)
        return json.dumps({"success": True, "data": predictions.tolist()})
    except Exception as e:
        return json.dumps({"success": False, "message": str(e)})

def perform_prediction(dataset, task, files):
    # Convert files to numpy arrays (assuming the uploaded files are in .npy format)
    data = [np.load(f) for f in files]
    # print("#############################")    
    # Model path selection based on dataset and task
    model_path = os.path.join(MODEL_DIR, dataset, f"bestmodel_{task}.hdf5")
    
    # Load model
    model = load_model(model_path, compile=False)
    
    # Make prediction
    output = model.predict_on_batch(data)
    return np.squeeze(output)

if __name__ == "__main__":
    print("Content-type: application/json\n")
    print(handle_file_upload())
