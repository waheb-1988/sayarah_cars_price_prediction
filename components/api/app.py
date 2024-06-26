import os
import sys
import numpy as np
from flask_cors import CORS, cross_origin
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(parent_dir)
from flask import Flask, request, jsonify
from components.models.preprocessing_list import predictions_1
import pickle 
import pathlib
import os  
import pandas as pd
import random

def generate_random_integer(start, end):
    return random.randint(start, end)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
dir_path = pathlib.Path(__file__).parent.parent
output = dir_path / "output"
path = dir_path / "data" 
dff= pd.read_csv(os.path.join(path, "data_lean.csv"))

with open("model.pkl", "rb") as file:
   model = pickle.load(file)

@app.route('/predict', methods=['POST'])

@cross_origin()

def predict():
    data = request.json
    random_number = random.randint(1, 1000) 
    dd=dff.iloc[[random_number]]
    predictions = int(model.predict(dd))
    result =  {"predictions" : str(np.round(predictions,0)),
               "status" : "ok",
                "response" : data}
    return result

if __name__ == '__main__':
    app.run()
