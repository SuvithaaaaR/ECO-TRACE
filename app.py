from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import joblib
import cv2
import numpy as np
import sqlite3
import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure the icons directory exists
icons_directory = os.path.join('static', 'icons')
os.makedirs(icons_directory, exist_ok=True)

# Create a default icon file
default_icon_path = os.path.join(icons_directory, 'default.png')
if not os.path.exists(default_icon_path):
    with open(default_icon_path, 'wb') as f:
        f.write(b'')  # Create an empty placeholder file

# Load model and labels
model = joblib.load("ewaste_model.pkl")
label_dict = joblib.load("label_dict.pkl")
reverse_labels = {v: k for k, v in label_dict.items()}

# Initialize database
DATABASE = 'ewaste.db'
conn = sqlite3.connect(DATABASE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS ewaste_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    classification TEXT,
    reward_points INTEGER
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS nonewaste_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    classification TEXT
)''')
conn.commit()

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read the image file: {image_path}")
    img = cv2.resize(img, (64, 64))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return hist.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Create tables if they don't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS ewaste_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        classification TEXT,
        reward_points INTEGER
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS nonewaste_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        classification TEXT
    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS schedule_pickups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        address TEXT,
        contact TEXT,
        ewaste_item TEXT,
        schedule TEXT
    )''')
    
    # Fetch all data for rendering the page
    try:
        schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
    except:
        schedule_data = []
    
    # Fetch unique e-waste classifications to display in dropdown
    try:
        ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
    except:
        ewaste_classifications = []
    
    try:
        ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
    except:
        ewaste_items = []
    
    try:
        nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
    except:
        nonewaste_items = []
    
    try:
        total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
    except:
        total_rewards = 0
    
    # Handle POST requests for filtering e-waste/non-e-waste
    filter_type = None
    if request.method == 'POST' and 'filter' in request.form:
        filter_type = request.form['filter']
    
    return render_template('index.html', 
                          schedule_data=schedule_data,
                          ewaste_items=ewaste_items,
                          ewaste_classifications=ewaste_classifications,
                          nonewaste_items=nonewaste_items,
                          total_rewards=total_rewards,
                          filter_type=filter_type)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict the uploaded image
        try:
            features = extract_features(filepath).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]
            threshold = 0.5

            # Sort probabilities in descending order
            sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

            if sorted_probabilities[0][1] < threshold:
                result = "Not an e-waste"
                classification = "Non-E-Waste"
                
                # Save to non-e-waste table
                cursor.execute('''INSERT INTO nonewaste_uploads (filename, classification) VALUES (?, ?)''',
                              (file.filename, classification))
            else:
                predicted_class = reverse_labels[sorted_probabilities[0][0]]
                result = f"This is an e-waste: {predicted_class} ({sorted_probabilities[0][1] * 100:.2f}%)"
                classification = predicted_class
                reward_points = 10  # Assign reward points for e-waste uploads
                
                # Save to e-waste table
                cursor.execute('''INSERT INTO ewaste_uploads (filename, classification, reward_points) VALUES (?, ?, ?)''',
                              (file.filename, classification, reward_points))
            
            conn.commit()
            
            # Fetch data for rendering the page
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html',
                                  result=result,
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)
                                  
        except Exception as e:
            return render_template('index.html', error=f"Error processing the image: {e}")

@app.route('/pickup', methods=['GET', 'POST'])
def pickup_request():
    if request.method == 'POST':
        try:
            name = request.form['name']
            address = request.form['address']
            contact = request.form['contact']
            ewaste_item = request.form['ewaste']
            schedule = request.form['schedule']

            # Save pickup request to database
            cursor.execute('''CREATE TABLE IF NOT EXISTS schedule_pickups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                address TEXT,
                contact TEXT,
                ewaste_item TEXT,
                schedule TEXT
            )''')
            cursor.execute('''INSERT INTO schedule_pickups (name, address, contact, ewaste_item, schedule) VALUES (?, ?, ?, ?, ?)''',
                           (name, address, contact, ewaste_item, schedule))
            conn.commit()

            # Fetch all data for rendering the page
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html', 
                                  message="Pickup request scheduled successfully!",
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)
        except KeyError as e:
            # Fetch all data for rendering the page
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html', 
                                  error=f"Missing form field: {e}",
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)

    # Fetch all data for rendering the page
    schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
    ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
    ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
    nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
    total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
    
    return render_template('index.html', 
                          schedule_data=schedule_data,
                          ewaste_items=ewaste_items,
                          ewaste_classifications=ewaste_classifications,
                          nonewaste_items=nonewaste_items,
                          total_rewards=total_rewards)

@app.route('/schedule', methods=['GET', 'POST'])
def schedule_pickup():
    if request.method == 'POST':
        try:
            name = request.form['name']
            address = request.form['address']
            contact = request.form['contact']
            ewaste_item = request.form['ewaste']
            schedule = request.form['schedule']

            # Create the schedule_pickups table if it doesn't exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS schedule_pickups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                address TEXT,
                contact TEXT,
                ewaste_item TEXT,
                schedule TEXT
            )''')
            
            # Insert the pickup details into the database
            cursor.execute('''INSERT INTO schedule_pickups (name, address, contact, ewaste_item, schedule) VALUES (?, ?, ?, ?, ?)''',
                         (name, address, contact, ewaste_item, schedule))
            conn.commit()
            
            # Fetch all scheduled pickups to display on the page
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html', 
                                  message="Pickup scheduled successfully!", 
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)
                                  
        except Exception as e:
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html', 
                                  error=f"Error scheduling pickup: {e}", 
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)

    # GET request - just display the page with existing data
    schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
    ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
    ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
    nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
    total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
    
    return render_template('index.html', 
                          schedule_data=schedule_data,
                          ewaste_items=ewaste_items,
                          ewaste_classifications=ewaste_classifications,
                          nonewaste_items=nonewaste_items,
                          total_rewards=total_rewards)

@app.route('/locator')
def dropoff_locator():
    return render_template('locator.html')

@app.route('/dashboard')
def user_dashboard():
    cursor.execute('''SELECT filename, classification, reward_points FROM uploads''')
    uploads = cursor.fetchall()
    total_rewards = sum(upload[2] for upload in uploads)
    return render_template('dashboard.html', uploads=uploads, total_rewards=total_rewards)

@app.route('/calculator', methods=['GET', 'POST'])
def e_waste_calculator():
    if request.method == 'POST':
        # Handle calculator logic here
        return render_template('calculator.html', result="Estimated impact calculated!")
    return render_template('calculator.html')

@app.route('/donation')
def donation_options():
    return render_template('donation.html')

@app.route('/services', methods=['GET', 'POST'])
def services():
    if request.method == 'POST':
        # Handle schedule picker logic here
        return render_template('services.html', message="Schedule picker added successfully!")
    return render_template('services.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
        ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
        ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
        nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
        total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
        
        return render_template('index.html', 
                              error="No file part",
                              schedule_data=schedule_data,
                              ewaste_items=ewaste_items,
                              ewaste_classifications=ewaste_classifications,
                              nonewaste_items=nonewaste_items,
                              total_rewards=total_rewards)
    
    file = request.files['image']
    if file.filename == '':
        schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
        ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
        ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
        nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
        total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
        
        return render_template('index.html', 
                              error="No selected file",
                              schedule_data=schedule_data,
                              ewaste_items=ewaste_items,
                              ewaste_classifications=ewaste_classifications,
                              nonewaste_items=nonewaste_items,
                              total_rewards=total_rewards)
    
    if file:
        # Create the upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image with our model
        try:
            # Extract features
            def extract_features(image_path):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"File not found: {image_path}")

                # Use cv2.imdecode to handle special characters in file paths
                with open(image_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError(f"Unable to read the image file: {image_path}")

                img = cv2.resize(img, (64, 64))
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                return hist.flatten()
            
            features = extract_features(filepath).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]
            threshold = 0.5
            
            # Sort probabilities in descending order
            sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
            
            # Create the necessary tables if they don't exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS ewaste_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                classification TEXT,
                reward_points INTEGER
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS nonewaste_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                classification TEXT
            )''')
            
            if sorted_probabilities[0][1] < threshold:
                result = "Not an e-waste"
                classification = "Non-E-Waste"
                
                # Save to non-e-waste table
                cursor.execute('''INSERT INTO nonewaste_uploads (filename, classification) VALUES (?, ?)''',
                              (file.filename, classification))
            else:
                predicted_class = reverse_labels[sorted_probabilities[0][0]]
                result = f"This is an e-waste: {predicted_class} ({sorted_probabilities[0][1] * 100:.2f}%)"
                classification = predicted_class
                reward_points = 10  # Assign reward points for e-waste uploads
                
                # Save to e-waste table
                cursor.execute('''INSERT INTO ewaste_uploads (filename, classification, reward_points) VALUES (?, ?, ?)''',
                              (file.filename, classification, reward_points))
            
            conn.commit()
            
            # Fetch data for rendering the page
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html',
                                  result=result,
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)
            
        except Exception as e:
            schedule_data = cursor.execute('''SELECT * FROM schedule_pickups''').fetchall()
            ewaste_classifications = cursor.execute('''SELECT DISTINCT classification FROM ewaste_uploads''').fetchall()
            ewaste_items = cursor.execute('''SELECT filename, classification FROM ewaste_uploads''').fetchall()
            nonewaste_items = cursor.execute('''SELECT filename, classification FROM nonewaste_uploads''').fetchall()
            total_rewards = cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''').fetchone()[0] or 0
            
            return render_template('index.html', 
                                  error=f"Error processing the image: {e}",
                                  schedule_data=schedule_data,
                                  ewaste_items=ewaste_items,
                                  ewaste_classifications=ewaste_classifications,
                                  nonewaste_items=nonewaste_items,
                                  total_rewards=total_rewards)
    
    return render_template('index.html', error="Something went wrong")

@app.route('/collected', methods=['GET'])
def collected_e_waste():
    cursor.execute('''SELECT filename, classification FROM uploads''')
    collected_items = cursor.fetchall()
    return render_template('collected.html', collected_items=collected_items)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/rewards', methods=['GET'])
def reward_system():
    cursor.execute('''SELECT SUM(reward_points) FROM ewaste_uploads''')
    total_rewards = cursor.fetchone()[0] or 0
    return render_template('index.html', total_rewards=total_rewards)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            message = request.form['message']
            
            # Create contact messages table if it doesn't exist
            cursor.execute('''CREATE TABLE IF NOT EXISTS contact_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                message TEXT,
                timestamp TEXT
            )''')
            
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save message to database
            cursor.execute('''INSERT INTO contact_messages (name, email, message, timestamp) VALUES (?, ?, ?, ?)''',
                        (name, email, message, timestamp))
            conn.commit()
            
            return render_template('index.html', message="Thank you for your message! We will get back to you soon.")
        except Exception as e:
            return render_template('index.html', error=f"Error sending message: {e}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)