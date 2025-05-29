from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error
import os
import numpy as np
from utils.preprocessor import preprocess_image
import logging
import uuid
from datetime import datetime
from functools import wraps
import torch
import torch.nn.functional as F
from src.model.train import get_model
import json
import secrets
from dotenv import load_dotenv

# Load environment variables from .flaskenv file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security configuration
app.config['SECRET_KEY'] = secrets.token_hex(16)

# File upload configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
MAX_FILE_SIZE_MB = 16
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'dcm'}
app.config['UPLOAD_EXTENSIONS_MESSAGE'] = f"Allowed file types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'chest_xray'
}

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'v1_pre.pth')
MODEL_ARCH = 'v1'  # Explicitly set architecture to match your model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
DISEASE_LABELS = ['Effusion', 'Mass', 'Nodule', 'No Finding']

# Ensure upload directory exists
upload_dir = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
app.config['UPLOAD_FOLDER'] = upload_dir

# Database connection function
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    try:
        # Get model architecture from filename
        arch = MODEL_PATH.split('/')[-1].split('_')[0]
        logger.info(f"Loading {arch} model from {MODEL_PATH}")
        
        model = get_model(arch, num_classes=NUM_CLASSES)
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Replace @app.before_first_request with this pattern
def load_model_on_startup():
    global model
    try:
        logger.info(f"Loading {MODEL_ARCH} model from {MODEL_PATH}")
        
        # Use explicit architecture instead of parsing from filename
        model = get_model(MODEL_ARCH, num_classes=NUM_CLASSES)
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

# Initialize model at startup
model = None
with app.app_context():
    load_model_on_startup()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form.get('login_id')  # Can be email or phone
        password = request.form.get('password')
        
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                # Check if login_id is email or phone
                cursor.execute("""
                    SELECT * FROM users 
                    WHERE email = %s OR phone_number = %s
                """, (login_id, login_id))
                
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['id']
                    session['name'] = f"{user['fname']} {user['lname']}"
                    return redirect(url_for('index'))
                    
                flash('Invalid credentials')
            except Error as e:
                flash('Login error occurred')
                logger.error(f"Login error: {e}")
            finally:
                cursor.close()
                conn.close()
                
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fname = request.form.get('fname')
        lname = request.form.get('lname')
        email = request.form.get('email')
        phone_number = request.form.get('phone_number')
        age = request.form.get('age')
        gender = request.form.get('gender')
        password = request.form.get('password')
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (fname, lname, email, phone_number, age, gender, password)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (fname, lname, email, phone_number, age, gender, hashed_password))
                
                conn.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
                
            except Error as e:
                conn.rollback()
                if 'Duplicate entry' in str(e):
                    flash('Email or phone number already registered')
                else:
                    flash('Registration error occurred')
                logger.error(f"Registration error: {e}")
            finally:
                cursor.close()
                conn.close()
                
    return render_template('signup.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    doctor_name = request.form.get('doctor_name')
    
    if not doctor_name:
        return jsonify({'error': 'Doctor name is required'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Fix the filepath handling
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = filename  # Store only filename in database
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(full_path)
        
        try:
            # Load and preprocess image
            img = preprocess_image(full_path)
            img_tensor = torch.from_numpy(img).float().to(DEVICE)  # Remove extra unsqueeze
            
            # Use global model instead of loading for each request
            if model is None:
                return jsonify({'error': 'Model not initialized'}), 500
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                
                # Get all predictions with confidence scores
                predictions = {
                    DISEASE_LABELS[i]: float(probabilities[i])
                    for i in range(len(DISEASE_LABELS))
                }
                
                # Sort predictions by confidence
                sorted_predictions = dict(
                    sorted(predictions.items(), 
                          key=lambda x: x[1], 
                          reverse=True)
                )
                
                max_disease = list(sorted_predictions.keys())[0]
                confidence = sorted_predictions[max_disease]
                
                # Save to database with more detailed result
                prediction_result = {
                    'primary': max_disease,
                    'confidence': confidence,
                    'all_predictions': sorted_predictions
                }
                
                # Save prediction to database
                conn = get_db_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO predictions 
                            (user_id, image_path, prediction_result, confidence, doctor_name)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            session['user_id'], 
                            clean_image_path(filename),
                            json.dumps(prediction_result),
                            confidence, 
                            doctor_name
                        ))
                        conn.commit()
                        prediction_id = cursor.lastrowid  # Get the ID of the inserted prediction

                        return jsonify({
                            'predictions': sorted_predictions,
                            'most_likely': max_disease,
                            'confidence': confidence,
                            'image_path': filename,
                            'prediction_id': prediction_id  # Add this line
                        })
                        
                    finally:
                        cursor.close()
                        conn.close()
                
                return jsonify({
                    'predictions': sorted_predictions,
                    'most_likely': max_disease,
                    'confidence': confidence,
                    'image_path': filename  # Return only filename
                })
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/history')
@login_required
def history():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE user_id = %s And rejected = 0
                ORDER BY timestamp DESC
            """, (session['user_id'],))
            
            predictions = cursor.fetchall()
            # Parse JSON strings
            for prediction in predictions:
                prediction['prediction_result'] = json.loads(prediction['prediction_result'])
            
            return render_template('history.html', predictions=predictions)
        finally:
            cursor.close()
            conn.close()
    
    return render_template('history.html', predictions=[])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/reject_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def reject_prediction(prediction_id):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            # Verify the prediction belongs to the current user
            cursor.execute("""
                UPDATE predictions 
                SET rejected = 1 
                WHERE id = %s AND user_id = %s
            """, (prediction_id, session['user_id']))
            
            if cursor.rowcount > 0:
                conn.commit()
                return jsonify({'success': True})
            else:
                return jsonify({'error': 'Prediction not found or unauthorized'}), 403
                
        except Error as e:
            logger.error(f"Error rejecting prediction: {e}")
            return jsonify({'error': 'Database error'}), 500
        finally:
            cursor.close()
            conn.close()
    
    return jsonify({'error': 'Database connection error'}), 500

# Add a function to clean image paths
def clean_image_path(filepath):
    """Clean image path to ensure consistent format"""
    return filepath.replace('\\', '/').replace('//', '/')

if __name__ == '__main__':
    app.run(debug=True)