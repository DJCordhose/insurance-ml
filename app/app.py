#!/usr/bin/env python

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys 
sys.path.append('../lib')    
# print(sys.path)

# from sklearn_prediction import predict
# from tf_prediction import predict
from model_prediction import predict

from prometheus_client import MetricsHandler, Counter

app = Flask(__name__)
CORS(app)

import logging

logging.basicConfig(filename='req.log')

data_logger = logging.getLogger('DataLogger')
data_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('data.log')
data_logger.addHandler(file_handler)

@app.route("/ping")
def ping():
    return "pong"

@app.route('/predict', methods=['GET', 'POST'])
def do_predict():
    speed = request.json['speed']
    age = request.json['age']
    miles = request.json['miles']

    try:
        predicted_category, probabilities = predict(speed, age, miles)

        response = {
            'category': predicted_category,
            'prediction': probabilities,
        }
        
        dataset = {
            'out': response,
            'in': {
                'speed': speed, 'age': age, 'miles': miles
            }
        }

        data_logger.info(dataset)
        return jsonify(response)
    except (ValueError):
        return jsonify({'error': 'invalid input'}), 422

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

