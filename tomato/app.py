from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the trained model
model = load_model('static/models/cnn_potato_disease_model_1.h5')

# Define the class labels (should match model's output)
class_labels = [
    'TOMATO__Late_blight',
    'TOMATO__Leaf_Mold',
    'TOMATO__Septoria_leaf_spot',
    'TOMATO__Spider_mites Two-spotted_spider_mite'
]

# Simulate a user database (optional expansion later)
users = []

# Route: Welcome page (Sign In)
@app.route("/")
def index():
    return render_template("signin.html")

# Route: Signup page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        users.append({"username": username, "password": password})
        print(f"User {username} signed up successfully.")

        return redirect(url_for("home"))

    return render_template("login.html")

# Route: Home page
@app.route("/home")
def home():
    return render_template("home.html")

# Route: Disease Prediction
@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'POST':
        crop = request.form.get('crop')
        file = request.files.get('image')
        
        if file:
            img_filename = f'{uuid.uuid4()}.jpg'
            img_path = os.path.join('static', img_filename)
            file.save(img_path)

            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_label = class_labels[predicted_class[0]]

            # Move this into Python instead of JS
            disease_info = {
                'TOMATO__Late_blight': {
                    'fertilizer': 'Use Mancozeb 75% WP at 2.5g/litre of water',
                    'precaution': 'Avoid overhead watering. Ensure proper spacing and drainage.'
                },
                'TOMATO__Leaf_Mold': {
                    'fertilizer': 'Apply Copper Oxychloride 50% WP @ 3g/litre',
                    'precaution': 'Remove infected leaves and improve air circulation in the crop canopy.'
                },
                'TOMATO__Septoria_leaf_spot': {
                    'fertilizer': 'Use Chlorothalonil 75% WP at 2g/litre of water',
                    'precaution': 'Rotate crops and do not work when plants are wet.'
                },
                'TOMATO__Spider_mites Two-spotted_spider_mite': {
                    'fertilizer': 'Apply Abamectin 1.8% EC @ 0.5 ml/litre',
                    'precaution': 'Use insecticidal soap and maintain humidity.'
                }
            }

            fertilizer = disease_info.get(predicted_label, {}).get('fertilizer', 'No data')
            precaution = disease_info.get(predicted_label, {}).get('precaution', 'No data')

            return render_template('disease_prediction.html',
                                   prediction=predicted_label,
                                   img_path=img_filename,
                                   fertilizer=fertilizer,
                                   precaution=precaution)
    return render_template('disease_prediction.html', prediction=None, img_path=None)

# Route: Chatbot
@app.route('/Chatbot')
def Chatbot():
    return render_template('chatbot.html')

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
