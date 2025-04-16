#!C:/Users/Admin/AppData/Local/Programs/Python/Python311/python.exe

import cgi
import cgitb
import os
import sys
import numpy as np
import tensorflow as tf
from keras.utils import get_custom_objects
import time
import random
import string
import pandas as pd
import io
import base64

# Set encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

cgitb.enable()

print("Content-Type: text/html; charset=utf-8\n")

# Add HTML styling with beautified version
print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCR-ESM Prediction Results</title>
    <style>
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --light-bg: #f8f9fa;
            --dark-bg: #343a40;
            --border-radius: 8px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        header {
            margin-bottom: 30px;
            border-bottom: 2px solid var(--light-bg);
            padding-bottom: 20px;
        }
        
        h1 {
            color: var(--yellow-color);
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        h2 {
            color: var(--secondary-color);
            font-size: 1.8rem;
            margin: 20px 0 15px;
        }
        
        h3 {
            color: var(--secondary-color);
            font-size: 1.4rem;
            margin: 15px 0 10px;
        }
        
        .result-container {
            margin: 30px 0;
        }
        
        .result {
            background-color: var(--light-bg);
            padding: 20px;
            border-radius: var(--border-radius);
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .result-item {
            background-color: white;
            border: 1px solid #e9ecef;
            border-radius: var(--border-radius);
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .prediction-score {
            font-size: 1.2rem;
            font-weight: bold;
            color: var(--primary-color);
            background-color: rgba(52, 152, 219, 0.1);
            padding: 15px;
            border-radius: var(--border-radius);
            margin-top: 10px;
            text-align: center;
        }
        
        .error {
            color: white;
            background-color: var(--danger-color);
            padding: 15px;
            border-radius: var(--border-radius);
            margin: 15px 0;
            box-shadow: var(--box-shadow);
        }
        
        .error h2 {
            color: white;
            margin-top: 0;
        }
        
        .success {
            color: white;
            background-color: var(--success-color);
            padding: 15px;
            border-radius: var(--border-radius);
            margin: 15px 0;
            box-shadow: var(--box-shadow);
        }
        
        .success h2 {
            color: white;
            margin-top: 0;
        }
        
        .debug {
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 10px;
            border-left: 4px solid #6c757d;
            margin: 5px 0;
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>TCR-ESM Prediction Results</h1>
        </header>
        <div class="result-container">
""")

try:
    # Initialize form
    form = cgi.FieldStorage()
    
    # Get database and task values
    database = form.getvalue("source", "").strip()  # mcpas or vdjdb
    task = form.getvalue("task", "").strip()        # one of 6 tasks

    # Debug information
    print(f'<div class="debug">Database: {database}</div>')
    print(f'<div class="debug">Task: {task}</div>')

    if not database or not task:
        print('<div class="error"><h2>Error:</h2><p>Please select both the database source and prediction task</p></div>')
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
        if name in form:
            try:
                # Create uploads directory if it doesn't exist
                upload_dir = "../htdocs/tcr-esm/uploads"
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                
                # Handle multiple files
                if isinstance(form[name], list):
                    arrays = []
                    for file_item in form[name]:
                        if hasattr(file_item, 'file') and file_item.file:
                            # Get original file extension
                            original_filename = file_item.filename
                            file_ext = os.path.splitext(original_filename)[1]
                            
                            # Generate unique filename
                            timestamp = int(time.time())
                            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                            unique_filename = f"{name}_{timestamp}_{random_str}{file_ext}"
                            
                            # Save the file
                            file_path = os.path.join(upload_dir, unique_filename)
                            with open(file_path, 'wb') as f:
                                f.write(file_item.file.read())
                            
                            # Load the numpy array
                            array = np.load(file_path)
                            # If array has multiple inputs (shape[0] > 1), split it
                            if len(array.shape) > 1 and array.shape[0] > 1:
                                arrays.extend([arr for arr in array])
                            else:
                                arrays.append(array)
                    return arrays if arrays else None
                else:
                    # Single file handling
                    if hasattr(form[name], 'file') and form[name].file:
                        original_filename = form[name].filename
                        file_ext = os.path.splitext(original_filename)[1]
                        timestamp = int(time.time())
                        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                        unique_filename = f"{name}_{timestamp}_{random_str}{file_ext}"
                        file_path = os.path.join(upload_dir, unique_filename)
                        with open(file_path, 'wb') as f:
                            f.write(form[name].file.read())
                        array = np.load(file_path)
                        # If array has multiple inputs (shape[0] > 1), split it
                        if len(array.shape) > 1 and array.shape[0] > 1:
                            return [arr for arr in array]
                        else:
                            return [array]
            except Exception as e:
                print(f'<div class="error"><h2>Error loading {name}:</h2><p>{str(e)}</p></div>')
                return None
        return None

    # Mapping task to required inputs
    task_inputs = {
        "alphaptide": ["tcra", "pep"],
        "betaptide": ["tcrb", "pep"],
        "alphaptidemhc": ["tcra", "pep", "mhc"],
        "betaptidemhc": ["tcrb", "pep", "mhc"],
        "alphabeta": ["tcra", "tcrb", "pep"],
        "alphabetaptidemhc": ["tcra", "tcrb", "pep", "mhc"]
    }

    # Load all input arrays
    tcra = load_array("tcra")
    tcrb = load_array("tcrb")
    pep = load_array("pep")
    mhc = load_array("mhc")

    # Debug information
    print(f'<div class="debug">Database: {database}</div>')
    print(f'<div class="debug">Task: {task}</div>')
    print(f'<div class="debug">TCRα inputs: {len(tcra) if tcra else 0}</div>')
    print(f'<div class="debug">TCRβ inputs: {len(tcrb) if tcrb else 0}</div>')
    print(f'<div class="debug">Peptide inputs: {len(pep) if pep else 0}</div>')
    print(f'<div class="debug">MHC inputs: {len(mhc) if mhc else 0}</div>')

    # Validate required inputs for the selected task
    required_inputs = task_inputs.get(task, [])
    if not required_inputs:
        print(f'<div class="error"><h2>Error:</h2><p>Invalid task type: {task}</p></div>')
        exit()

    # Check if all required inputs are provided
    missing_inputs = []
    for input_type in required_inputs:
        if input_type == "tcra" and not tcra:
            missing_inputs.append("TCRα")
        elif input_type == "tcrb" and not tcrb:
            missing_inputs.append("TCRβ")
        elif input_type == "pep" and not pep:
            missing_inputs.append("Peptide")
        elif input_type == "mhc" and not mhc:
            missing_inputs.append("MHC")

    if missing_inputs:
        print(f'<div class="error"><h2>Error:</h2><p>Missing required inputs: {", ".join(missing_inputs)}</p></div>')
        exit()

    # Check if we have multiple inputs
    is_multiple = any(isinstance(arr, list) and len(arr) > 1 for arr in [tcra, tcrb, pep, mhc])
    
    if is_multiple:
        # Ensure all required inputs have the same number of samples
        lengths = [len(arr) if isinstance(arr, list) else 1 for arr in [tcra, tcrb, pep, mhc]]
        required_lengths = [lengths[i] for i, arr in enumerate([tcra, tcrb, pep, mhc]) if arr is not None]
        if not all(l == required_lengths[0] for l in required_lengths):
            print('<div class="error"><h2>Error:</h2><p>Number of inputs must be the same for all required files</p></div>')
            exit()

        # Store predictions for Excel file and display
        predictions = []
        
        # Process each set of inputs
        for i in range(required_lengths[0]):
            # Prepare input data based on task requirements
            input_data = []
            for input_type in required_inputs:
                if input_type == "tcra":
                    arr = tcra[i] if isinstance(tcra, list) else tcra[0]
                    if len(arr.shape) == 1:
                        arr = np.expand_dims(arr, axis=0)
                    input_data.append(arr)
                elif input_type == "tcrb":
                    arr = tcrb[i] if isinstance(tcrb, list) else tcrb[0]
                    if len(arr.shape) == 1:
                        arr = np.expand_dims(arr, axis=0)
                    input_data.append(arr)
                elif input_type == "pep":
                    arr = pep[i] if isinstance(pep, list) else pep[0]
                    if len(arr.shape) == 1:
                        arr = np.expand_dims(arr, axis=0)
                    input_data.append(arr)
                elif input_type == "mhc":
                    arr = mhc[i] if isinstance(mhc, list) else mhc[0]
                    if len(arr.shape) == 1:
                        arr = np.expand_dims(arr, axis=0)
                    input_data.append(arr)

            # Make prediction
            prediction = model.predict(input_data)
            pred_score = prediction[0][0]
            predictions.append((i+1, pred_score))

        # Create Excel file
        df = pd.DataFrame(predictions, columns=['Input Number', 'Prediction Score'])
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            worksheet = writer.sheets['Predictions']
            worksheet.set_column('A:B', 15)  # Set column width
        excel_data = excel_buffer.getvalue()
        excel_base64 = base64.b64encode(excel_data).decode()

        
        # Start the results table
        print("""
        <div class="result">
            <h2>Prediction Results</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Sample</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Prediction Score</th>
                    </tr>
                </thead>
                <tbody>
        """)
        
        # Add rows to table
        for i, (sample_num, pred_score) in enumerate(predictions):
            print(f"""
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 12px;">{sample_num}</td>
                        <td style="padding: 12px;">{pred_score:.4f}</td>
                    </tr>
            """)
        
        # Close the table
        print("""
                </tbody>
            </table>
        </div>
        """)
    else:
        # Single input processing
        predictions = []  # Initialize predictions list
        input_data = []
        for input_type in required_inputs:
            if input_type == "tcra":
                arr = tcra[0]
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, axis=0)
                input_data.append(arr)
            elif input_type == "tcrb":
                arr = tcrb[0]
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, axis=0)
                input_data.append(arr)
            elif input_type == "pep":
                arr = pep[0]
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, axis=0)
                input_data.append(arr)
            elif input_type == "mhc":
                arr = mhc[0]
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, axis=0)
                input_data.append(arr)

        prediction = model.predict(input_data)
        pred_score = prediction[0][0]
        predictions.append((1, pred_score))

        # Create Excel file
        df = pd.DataFrame(predictions, columns=['Input Number', 'Prediction Score'])
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            worksheet = writer.sheets['Predictions']
            worksheet.set_column('A:B', 15)  # Set column width
        excel_data = excel_buffer.getvalue()
        excel_base64 = base64.b64encode(excel_data).decode()

        # Add download button
        print(f"""
        <div class="download-section" style="margin-top: 20px; text-align: center;">
            <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_base64}" 
               download="prediction_results.xlsx" 
               class="download-btn" 
               style="display: inline-block; 
                      padding: 10px 20px; 
                      background-color: #4CAF50; 
                      color: white; 
                      text-decoration: none; 
                      border-radius: 5px; 
                      font-weight: bold;">
                Download Results
            </a>
        </div>
        """)
        
        # Display single result in a table format
        print("""
        <div class="result">
            <h2>Prediction Results</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Sample</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Prediction Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 12px;">1</td>
                        <td style="padding: 12px;">{:.4f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """.format(pred_score))

except Exception as e:
    print(f'<div class="error"><h2>Error processing form:</h2><p>{str(e)}</p></div>')

print("""
        </div>
    </div>
</body>
</html>
""")
