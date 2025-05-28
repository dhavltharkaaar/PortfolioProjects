from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("cars_24_combined.csv")
data = data.dropna(subset=['Year'])
data['Year'] = data['Year'].astype(int)

model = pickle.load(open("carprice_model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    car_name = sorted(data['Car Name'].astype(str).unique())
    model_year = sorted(data['Year'].unique(), reverse=True)
    km_driven_values = sorted(data['Distance'].unique(), reverse=True)
    km_driven = int(max(km_driven_values))
    fuel_type = sorted(data['Fuel'].astype(str).unique())
    type_driven = sorted(data['Drive'].astype(str).unique())
    model_type = sorted(data['Type'].astype(str).unique())
    return render_template('Index.html', car_names=car_name, model_years=model_year, km_drivens=km_driven, fuel_types=fuel_type, type_drivens=type_driven, model_types=model_type)



@app.route('/predict', methods=['POST'])
def predict():
    car_name = request.form.get('cars_name')
    model_year = int(request.form.get('model_year'))
    km_driven = int(request.form.get('km_driven'))
    fuel_type = request.form.get('fuel_type')
    type_driven = request.form.get('type_driven')
    model_type = request.form.get('model_type')

    input_data = pd.DataFrame([[car_name, model_year, km_driven, fuel_type, type_driven, model_type]],
                               columns=['Car Name', 'Year', 'Distance', 'Fuel', 'Drive', 'Type'])

    input_data['Car Name'] = input_data['Car Name'].fillna('Unknown')  
    input_data['Fuel'] = input_data['Fuel'].fillna('Unknown') 
    input_data['Drive'] = input_data['Drive'].fillna('Unknown')  
    input_data['Type'] = input_data['Type'].fillna('Unknown')  

    input_data['Car Name'] = input_data['Car Name'].astype(str)
    input_data['Fuel'] = input_data['Fuel'].astype(str)
    input_data['Drive'] = input_data['Drive'].astype(str)
    input_data['Type'] = input_data['Type'].astype(str)

    try:
        prediction = model.predict(input_data)
        result = np.round(prediction[0], 2)
        return str(result)
    except Exception as e:
        return f"Error predicting car price: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
