# sheraj
import os
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as tf
import CNN
import numpy as np
import torch
import pandas as pd
import time
import gdown

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Model file configuration
MODEL_FILE = "plant_disease_model_1_latest.pt"
MODEL_URL = "https://drive.google.com/uc?id=1MYbBpWGLs7cwoabn6skWpEXnZjWvIAtP"  # Updated with your file ID

# Load model
def load_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model file...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    
    model = CNN.CNN(39)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model

model = load_model()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = tf.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        if image and image.filename:
            try:
                # Generate a unique filename
                filename = f"{int(time.time())}_{image.filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save the file
                image.save(file_path)
                print(f"File saved successfully at: {file_path}")
                
                # Make prediction
                pred = prediction(file_path)
                title = disease_info['disease_name'][pred]
                description = disease_info['description'][pred]
                prevent = disease_info['Possible Steps'][pred]
                image_url = disease_info['image_url'][pred]
                supplement_name = supplement_info['supplement name'][pred]
                supplement_image_url = supplement_info['supplement image'][pred]
                supplement_buy_link = supplement_info['buy link'][pred]
                
                return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                    image_url=image_url, pred=pred, sname=supplement_name, 
                                    simage=supplement_image_url, buy_link=supplement_buy_link)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return "Error processing the image. Please try again."
        else:
            return "No image file was uploaded."

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                         supplement_image=list(supplement_info['supplement image']),
                         supplement_name=list(supplement_info['supplement name']), 
                         disease=list(disease_info['disease_name']), 
                         buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
