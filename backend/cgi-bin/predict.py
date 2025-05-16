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
import shutil
import logging
from contextlib import redirect_stdout

# Suppress TensorFlow progress output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

cgitb.enable()

# Get form data first
form = cgi.FieldStorage()

# Get session_id or generate a new one
session_id = form.getvalue("session_id", "").strip()
if not session_id:
    timestamp = str(int(time.time()))
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    session_id = f"{timestamp}-{random_str}"

# Create user-specific directory for the session
base_upload_dir = "../tmp/tcr_esm_uploads"
if not os.path.exists(base_upload_dir):
    os.makedirs(base_upload_dir)

user_upload_dir = os.path.join(base_upload_dir, session_id)
if not os.path.exists(user_upload_dir):
    os.makedirs(user_upload_dir)

# Check if this is a redirect with session_id in query string
is_redirect = False
query_string = os.environ.get('QUERY_STRING', '')
if 'view_results=true' in query_string and 'session_id=' in query_string:
    is_redirect = True

# Process and save any uploaded files first before potential redirect
def save_uploaded_file(field_name):
    if field_name in form:
        fileitem = form[field_name]
        if fileitem.file and fileitem.filename:
            # Save the file to the user's directory
            temp_file_path = os.path.join(user_upload_dir, f"{field_name}_input.npy")
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(fileitem.file.read())
            return True
    return False

# If we have a new form submission with file uploads, save them before redirecting
if not is_redirect and 'source' in form and 'task' in form:
    # Save any uploaded files
    save_uploaded_file("tcra")
    save_uploaded_file("tcrb")
    save_uploaded_file("pep")
    save_uploaded_file("mhc")
    
    # Get database and task values 
    database = form.getvalue("source", "").strip()
    task = form.getvalue("task", "").strip()
    
    # Write a status file to indicate processing is complete
    with open(os.path.join(user_upload_dir, "uploads_complete.txt"), 'w') as f:
        f.write("1")
    
    # Redirect to results page with session ID
    print("Status: 302 Found")
    print(f"Location: predict.py?view_results=true&session_id={session_id}&database={database}&task={task}")
    print()
    sys.exit(0)

# For either a redirect or direct view, show the results
print("Content-Type: text/html; charset=utf-8\n")

# Add HTML styling with beautified version
print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCR-ESM Prediction Results</title>
    <link rel="icon" type="image/png" href="../htdocs/tcr-esm/images/iiitd_logo.png">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --black-color: #000000;
            --blue-color: #0000FF;
            --primary-color: #FFC107;
            --secondary-color: #f5f5f5;
            --background-color: #ffffff;
            --text-color: #333333;
            --hover-color: #FFA000;
            --border-color: #e0e0e0;
            --header-strip-color: #FFC107;
            --gradient-start: #FFC107;
            --gradient-end: #FF9800;
        }

        body {
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        .container {
            max-width: 95%;
            margin: 0 auto;
            padding: 2rem 0;
            width: 100%;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 3rem;
            background: linear-gradient(135deg, #256576, #2d7a8f);
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
            width: 100%;
            box-sizing: border-box;
            min-height: 200px;
        }

        .logo-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.15), rgba(255,255,255,0.1));
            z-index: 1;
        }

        .logo-container::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: radial-gradient(circle, var(--primary-color) 2px, transparent 2px);
            background-size: 30px 30px;
            background-position: 0 0, 15px 15px;
            opacity: 0.15;
            z-index: 0;
            pointer-events: none;
        }

        .main-title {
            font-size: clamp(2.5rem, 4.5vw, 3.5rem);
            font-weight: 700;
            margin: 0;
            color: #FBFCFD;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            line-height: 1.2;
            text-align: center;
            position: relative;
            z-index: 2;
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem 2rem;
            border-radius: 15px;
            backdrop-filter: blur(5px);
        }

        .results-section {
            background-color: white;
            border-radius: 20px;
            padding: 3rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 800px;
            margin: 0 auto;
        }

        .download-section {
            background-color: var(--secondary-color);
            padding: 2.5rem;
            border-radius: 15px;
            margin: 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .download-title {
            font-size: 2rem;
            color: #2d7a8f;
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .download-description {
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }

        .download-buttons {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }

        .download-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 2rem;
            background: linear-gradient(135deg, #256576, #2d7a8f);
            color: white;
            text-decoration: none;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            min-width: 200px;
            border: none;
            cursor: pointer;
        }

        .download-btn i {
            margin-right: 0.8rem;
            font-size: 1.2rem;
        }

        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            background: linear-gradient(135deg, #2d7a8f, #256576);
            color: white;
        }

        .download-btn:active {
            transform: translateY(0);
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 2rem;
            background-color: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        .results-table th {
            background: linear-gradient(135deg, #256576, #2d7a8f);
            color: white;
            padding: 1.2rem;
            text-align: left;
            font-weight: 600;
        }

        .results-table td {
            padding: 1.2rem;
            border-bottom: 1px solid var(--border-color);
        }

        .results-table tr:last-child td {
            border-bottom: none;
        }

        .results-table tr:hover {
            background-color: rgba(45, 122, 143, 0.05);
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .logo-container {
                padding: 2rem;
                min-height: 150px;
            }
            
            .main-title {
                font-size: 2rem;
            }
            
            .results-section {
                padding: 2rem;
            }
            
            .download-buttons {
                flex-direction: column;
                gap: 1rem;
            }
            
            .download-btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo-container">
            <h1 class="main-title">TCR-ESM Prediction Results</h1>
        </div>
        <div class="results-section">
        
                <tbody>
""")

try:
    # Get database and task values from query string if this is a redirect view
    if is_redirect:
        database = form.getvalue("database", "").strip()  # mcpas or vdjdb
        task = form.getvalue("task", "").strip()        # one of 6 tasks
    else:
        # Get values from form if not a redirect
        database = form.getvalue("source", "").strip()  # mcpas or vdjdb
        task = form.getvalue("task", "").strip()        # one of 6 tasks

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

    # For redirected requests, check if we have saved files in the user directory
    def check_saved_file(name):
        file_path = os.path.join(user_upload_dir, f"{name}_input.npy")
        if os.path.exists(file_path):
            try:
                arr = np.load(file_path, allow_pickle=True)
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        return [np.expand_dims(arr, axis=0)]
                    elif arr.ndim > 1:
                        if arr.shape[0] > 1:
                            return [arr[i:i+1] for i in range(arr.shape[0])]
                        else:
                            return [arr]
                return arr
            except Exception as e:
                print(f'<div class="error"><h2>Error loading saved {name} file:</h2><p>{e}</p></div>')
        return None

    # Helper function to load arrays from uploaded files or saved files
    def load_array(name):
        if name in form:
            try:
                fileitem = form[name]
                
                if not fileitem.file:
                    return check_saved_file(name)
                
                temp_file_path = os.path.join(user_upload_dir, f"{name}_input.npy")
                
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(fileitem.file.read())
                
                arr = np.load(temp_file_path, allow_pickle=True)
                
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 1:
                        return [np.expand_dims(arr, axis=0)]
                    elif arr.ndim > 1:
                        if arr.shape[0] > 1:
                            return [arr[i:i+1] for i in range(arr.shape[0])]
                        else:
                            return [arr]
                return arr
                
            except Exception as e:
                return check_saved_file(name)
        
        return check_saved_file(name)

    # Load all input arrays
    tcra = load_array("tcra")
    tcrb = load_array("tcrb")
    pep = load_array("pep")
    mhc = load_array("mhc")

    # Check if we have multiple inputs
    is_multiple = False
    for arr in [tcra, tcrb, pep, mhc]:
        if arr is not None:
            if isinstance(arr, list) and len(arr) > 1:
                is_multiple = True
                break
            elif isinstance(arr, np.ndarray) and arr.ndim > 1 and arr.shape[0] > 1:
                is_multiple = True
                break

    # Mapping task to required inputs
    task_inputs = {
        "alphaptide": ["tcra", "pep"],
        "betaptide": ["tcrb", "pep"],
        "alphaptidemhc": ["tcra", "pep", "mhc"],
        "betaptidemhc": ["tcrb", "pep", "mhc"],
        "alphabeta": ["tcra", "tcrb", "pep"],
        "alphabetaptidemhc": ["tcra", "tcrb", "pep", "mhc"]
    }

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

    if is_multiple:
        # Ensure all required inputs have the same number of samples
        sample_counts = []
        for input_type in required_inputs:
            if input_type == "tcra" and tcra:
                sample_counts.append(len(tcra) if isinstance(tcra, list) else tcra.shape[0])
            elif input_type == "tcrb" and tcrb:
                sample_counts.append(len(tcrb) if isinstance(tcrb, list) else tcrb.shape[0])
            elif input_type == "pep" and pep:
                sample_counts.append(len(pep) if isinstance(pep, list) else pep.shape[0])
            elif input_type == "mhc" and mhc:
                sample_counts.append(len(mhc) if isinstance(mhc, list) else mhc.shape[0])
        
        if not sample_counts or not all(count == sample_counts[0] for count in sample_counts):
            print('<div class="error"><h2>Error:</h2><p>Number of inputs must be the same for all required files</p></div>')
            exit()
        
        num_samples = sample_counts[0]
        predictions = []
        
        # Process each set of inputs
        for i in range(num_samples):
            input_data = []
            for input_type in required_inputs:
                if input_type == "tcra":
                    if isinstance(tcra, list):
                        arr = tcra[i] if i < len(tcra) else tcra[0]
                    else:
                        arr = tcra[i:i+1]
                    input_data.append(arr)
                elif input_type == "tcrb":
                    if isinstance(tcrb, list):
                        arr = tcrb[i] if i < len(tcrb) else tcrb[0]
                    else:
                        arr = tcrb[i:i+1]
                    input_data.append(arr)
                elif input_type == "pep":
                    if isinstance(pep, list):
                        arr = pep[i] if i < len(pep) else pep[0]
                    else:
                        arr = pep[i:i+1]
                    input_data.append(arr)
                elif input_type == "mhc":
                    if isinstance(mhc, list):
                        arr = mhc[i] if i < len(mhc) else mhc[0]
                    else:
                        arr = mhc[i:i+1]
                    input_data.append(arr)

            # Make prediction with suppressed output
            with redirect_stdout(io.StringIO()):
                prediction = model.predict(input_data)
            pred_score = prediction[0][0]
            predictions.append((i+1, pred_score))

        # Create Excel file
        df = pd.DataFrame(predictions, columns=['Input Number', 'Prediction Score'])
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            worksheet = writer.sheets['Predictions']
            worksheet.set_column('A:B', 15)
        excel_data = excel_buffer.getvalue()
        excel_base64 = base64.b64encode(excel_data).decode()

        # Generate CSV content
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        csv_base64 = base64.b64encode(csv_data.encode()).decode()

    else:
        # Single input processing
        predictions = []
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

        # Make prediction with suppressed output
        with redirect_stdout(io.StringIO()):
            prediction = model.predict(input_data)
        pred_score = prediction[0][0]
        predictions.append((1, pred_score))

        # Create Excel file
        df = pd.DataFrame(predictions, columns=['Input Number', 'Prediction Score'])
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            worksheet = writer.sheets['Predictions']
            worksheet.set_column('A:B', 15)
        excel_data = excel_buffer.getvalue()
        excel_base64 = base64.b64encode(excel_data).decode()

        # Generate CSV content
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        csv_base64 = base64.b64encode(csv_data.encode()).decode()

    # Save results to files
    try:
        results_file_path = os.path.join(user_upload_dir, f"{session_id}_results.xlsx")
        csv_file_path = os.path.join(user_upload_dir, f"{session_id}_results.csv")
        
        with pd.ExcelWriter(results_file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
            worksheet = writer.sheets['Predictions']
            worksheet.set_column('A:B', 15)
        
        df.to_csv(csv_file_path, index=False)

        # Display results and download options
        print("""
        <div class="results-section">
            <div class="download-section">
                <h3 class="download-title">Download Your Results</h3>
                <p class="download-description">Your prediction results are ready. Download them in your preferred format:</p>
                <div class="download-buttons">
                    <a href="download.py?type=excel&session_id={session_id}" 
                       class="download-btn">
                        <i class="fas fa-file-excel"></i> Download Excel
                    </a>
                    <a href="download.py?type=csv&session_id={session_id}" 
                       class="download-btn">
                        <i class="fas fa-file-csv"></i> Download CSV
                    </a>
                </div>
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Sample</th>
                        <th>Prediction Score</th>
                    </tr>
                </thead>
                <tbody>
        """.format(session_id=session_id))

        # Add rows to table
        for i, (sample_num, pred_score) in enumerate(predictions):
            print(f"""
                    <tr>
                        <td>{sample_num}</td>
                        <td>{pred_score:.4f}</td>
                    </tr>
            """)

        # Close the table and sections
        print("""
                </tbody>
            </table>
        </div>
        """)

    except Exception as e:
        print(f'<div class="error"><h2>Error saving results:</h2><p>{e}</p></div>')

except Exception as e:
    print(f'<div class="error"><h2>Error processing form:</h2><p>{str(e)}</p></div>')

print("""
    </div>
</body>
</html>
""")
