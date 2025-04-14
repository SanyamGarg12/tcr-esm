#!C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe

import cgi
import cgitb
import os
import sys
import numpy as np
import tensorflow as tf
from keras.utils import get_custom_objects

# Set encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

cgitb.enable()

print("Content-Type: text/html; charset=utf-8\n")

# Add HTML styling
print("""
<!DOCTYPE html>
<html>
<head>
    <title>TCR-ESM Prediction Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #2c3e50;
        }
        .result {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success {
            color: #28a745;
            background-color: #d4edda;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
""")

try:
    # Initialize form
    form = cgi.FieldStorage()
    
    database = form.getvalue("source")  # mcpas or vdjdb
    task = form.getvalue("task")        # one of 6 tasks

    if not database or not task:
        print('<div class="error"><h2>Error:</h2><p>Missing required form fields (source or task)</p></div>')
        exit()

    # Mapping task to model file names
    model_mapping = {
        "alphaptide": "bestmodel_alphapeptide.hdf5",
        "betaptide": "bestmodel_betapeptide.hdf5",
        "alphaptidemhc": "bestmodel_alphapeptidemhc.hdf5",
        "betaptidemhc": "bestmodel_betapeptidemhc.hdf5",
        "alphabeta": "bestmodel_alphabetapeptide.hdf5",
        "alphabetaptidemhc": "bestmodel_alphabetapeptidemhc.hdf5"
    }

    # Define custom metric (placeholder MCC)
    def keras_mcc(y_true, y_pred):
        y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos
        y_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = tf.reduce_sum(y_pos * y_pred_pos)
        tn = tf.reduce_sum(y_neg * y_pred_neg)
        fp = tf.reduce_sum(y_neg * y_pred_pos)
        fn = tf.reduce_sum(y_pos * y_pred_neg)

        numerator = (tp * tn) - (fp * fn)
        denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7)
        return numerator / (denominator + 1e-7)

    # Register custom metric
    get_custom_objects().update({'keras_mcc': keras_mcc})

    # Determine model path
    model_name = model_mapping.get(task)
    if not model_name:
        print(f'<div class="error"><h2>Invalid Task:</h2><p>{task}</p></div>')
        exit()

    model_path = f"models/{database}/{model_name}"

    # Try loading model
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'keras_mcc': keras_mcc})
    except Exception as e:
        print(f'<div class="error"><h2>Error loading model:</h2><p>{e}</p></div>')
        exit()

    # Load input arrays
    def load_array(name):
        if name in form and form[name].file:
            try:
                # Create uploads directory if it doesn't exist
                upload_dir = "../htdocs/tcr-esm/uploads"
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                
                # Generate unique filename
                import time
                import random
                import string
                
                # Get original file extension
                original_filename = form[name].filename
                file_ext = os.path.splitext(original_filename)[1]
                
                # Generate unique filename with timestamp and random string
                timestamp = int(time.time())
                random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                unique_filename = f"{name}_{timestamp}_{random_str}{file_ext}"
                
                # Save the file
                file_path = os.path.join(upload_dir, unique_filename)
                with open(file_path, 'wb') as f:
                    f.write(form[name].file.read())
                
                # Load the numpy array
                return np.load(file_path)
            except Exception as e:
                print(f'<div class="error"><h2>Error loading {name}:</h2><p>{str(e)}</p></div>')
                return None
        return None

    # Load only required files based on task
    tcra = None
    tcrb = None
    peptide = None
    mhc = None

    if task == "alphaptide":
        tcra = load_array("tcra")
        peptide = load_array("pep")
        if tcra is None or peptide is None:
            print('<div class="error"><h2>Error:</h2><p>Both TCRα and Peptide arrays are required for alphaptide task</p></div>')
            exit()
        inputs = [tcra, peptide]
    elif task == "betaptide":
        tcrb = load_array("tcrb")
        peptide = load_array("pep")
        if tcrb is None or peptide is None:
            print('<div class="error"><h2>Error:</h2><p>Both TCRβ and Peptide arrays are required for betaptide task</p></div>')
            exit()
        inputs = [tcrb, peptide]
    elif task == "alphaptidemhc":
        tcra = load_array("tcra")
        peptide = load_array("pep")
        mhc = load_array("mhc")
        if tcra is None or peptide is None or mhc is None:
            print('<div class="error"><h2>Error:</h2><p>TCRα, Peptide, and MHC arrays are required for alphaptidemhc task</p></div>')
            exit()
        inputs = [tcra, peptide, mhc]
    elif task == "betaptidemhc":
        tcrb = load_array("tcrb")
        peptide = load_array("pep")
        mhc = load_array("mhc")
        if tcrb is None or peptide is None or mhc is None:
            print('<div class="error"><h2>Error:</h2><p>TCRβ, Peptide, and MHC arrays are required for betaptidemhc task</p></div>')
            exit()
        inputs = [tcrb, peptide, mhc]
    elif task == "alphabeta":
        tcra = load_array("tcra")
        tcrb = load_array("tcrb")
        peptide = load_array("pep")
        if tcra is None or tcrb is None or peptide is None:
            print('<div class="error"><h2>Error:</h2><p>TCRα, TCRβ, and Peptide arrays are required for alphabeta task</p></div>')
            exit()
        inputs = [tcra, tcrb, peptide]
    elif task == "alphabetaptidemhc":
        tcra = load_array("tcra")
        tcrb = load_array("tcrb")
        peptide = load_array("pep")
        mhc = load_array("mhc")
        if tcra is None or tcrb is None or peptide is None or mhc is None:
            print('<div class="error"><h2>Error:</h2><p>TCRα, TCRβ, Peptide, and MHC arrays are required for alphabetaptidemhc task</p></div>')
            exit()
        inputs = [tcra, tcrb, peptide, mhc]

    # Make prediction
    try:
        prediction = model.predict(inputs)
        print('<div class="result">')
        print('<h2>Prediction Results</h2>')
        print('<p>Task: ' + task + '</p>')
        print('<p>Database: ' + database + '</p>')
        print('<p>Prediction Score: ' + str(prediction[0][0]) + '</p>')
        print('</div>')
    except Exception as e:
        print(f'<div class="error"><h2>Prediction Error:</h2><p>{e}</p></div>')

except Exception as e:
    print(f'<div class="error"><h2>Error processing form:</h2><p>{str(e)}</p></div>')

print("""
    </div>
</body>
</html>
""")
