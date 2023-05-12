import os
import sys
import numpy as np
from flask_cors import CORS, cross_origin
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(parent_dir)
from flask import Flask, request, jsonify
from components.models.preprocessing import predictions_1
import pickle 
import pathlib
import os  
import pandas as pd
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
dir_path = pathlib.Path(__file__).parent.parent
output = dir_path / "output"
path = dir_path / "data" 
file_name = "data.csv"

with open("model.pkl", "rb") as file:
   model = pickle.load(file)

@app.route('/predict', methods=['POST'])

@cross_origin()

def predict():
    data = request.json
    print(data)
    dd = pd.DataFrame(data)
    print("adddd")
    print(dd)
    data_in= predictions_1.(dd)
    predictions = model.predict(data_in)

    return str(np.round(predictions,2))

if __name__ == '__main__':
    app.run()
