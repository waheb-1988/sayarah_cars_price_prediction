import os
import sys
from flask_cors import CORS, cross_origin
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(parent_dir)
from flask import Flask, request, jsonify
from components.models.preprocessing_list import predictions_1
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
df= pd.read_csv(os.path.join(path, file_name))

@app.route('/predict', methods=['POST'])

@cross_origin()

def predict():
    data = request.json
    dd = pd.DataFrame(data)
    print("###################")
    print(dd)
    pred=predictions_1(df)
    #mm,cc=pred.preprocess_inputs(dd)
    ff = pred.prrr(dd)
    
    response = {'predictions': ff}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
