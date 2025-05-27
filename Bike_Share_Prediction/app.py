# import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('lr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
   return render_template('contact.html')

@app.route('/predict',methods =['GET','POST'])
def predict():
    if request.method == 'POST':
        season = int(request.form['season'])
        month = int(request.form['month'])
        hr = int(request.form['hr'])
        holiday = int(request.form['holiday'])
        weekday = int(request.form['weekday'])
        workingday = int(request.form['workingday'])
        wheather_condition = float(request.form.get('wheather_condition', False))
        temp = int(request.form['temp'])
        atemp = int(request.form['atemp'])
        humidity = int(request.form['humidity'])
        windspeed = int(request.form['windspeed'])
        casual = int(request.form['casual'])
        registered = int(request.form['registered'])
        prediction = model.predict([[season,month,hr,holiday,weekday,workingday,wheather_condition,temp,atemp,humidity,windspeed,casual,registered]])
        output=round(prediction[0])
        if output<=0:
            return render_template('prediction.html',prediction_texts="Sorry bike not available")
        else:
            return render_template('prediction.html',prediction_text="Number of Bike available for sharing : {}".format(output))
    else:
        return render_template('prediction.html')


if __name__ == "__main__":
    app.run(debug=True)
