from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset to extract unique location names
df = pd.read_csv('C:/Real-Estate-Price-Prediction-main/Bengaluru_House_Data.csv')
location_names = df['location'].unique()

# Create location mapping
location_mapping = {location: idx for idx, location in enumerate(location_names)}

@app.route('/')
def index():
    return render_template('index.html', locations=location_names)

@app.route('/about')
def about():
    return render_template('about.html',locations=location_names)
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/property')
def prportey():
    return render_template('property.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input from the form
        location_name = request.form['location']
        size = int(request.form['size'])
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])

        # Convert location name to encoded value
        location = location_mapping.get(location_name)

        # Make prediction
        input_data = (location, size, total_sqft, bath)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)[0]  # Get the first prediction

        # Convert prediction to lakhs and round to 2 decimal places
        prediction_in_lakhs = round(prediction / 10, 2)

        # Get coordinates for the location
        coordinates = get_coordinates(location_name)

        return render_template('property.html', prediction=f"{prediction_in_lakhs} lakhs", location=location_name, coordinates=coordinates)



def get_coordinates(location_name):
    # Use Google Maps Geocoding API to get coordinates
    api_key = 'AIzaSyBOdO6I6le_gaZQsopYn9XS_ohCufGo7Vs'
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={location_name}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        latitude = location['lat']
        longitude = location['lng']
        return latitude, longitude
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)   