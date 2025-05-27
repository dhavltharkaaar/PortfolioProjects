import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Month':1, 'Hour':1, 'Weekday':6,'Registered_Users':32}

print(r.json())
