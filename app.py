# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, request, jsonify, Response
import pickle
import pandas as pd
import json

app = Flask(__name__)

def load_models():
    model = pickle.load(open('models/LRModel2.obj','rb'))
    return model

def load_data():
    data = pd.read_csv('./df_final_sample.csv')
    return data

def filter_dataset(Id,data):
    'Filters dataset down to a single line, being the chosen client ID'
    df_small = data[data['SK_ID_CURR'] == Id]
    df_small.drop(columns = ['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], inplace = True)
    return df_small


@app.route('/test_api/', methods=['GET'])
def test_pred():
    Id = int(request.args["Id"])
    print(Id)
    try : 
        try:
            data = load_data()
        except:
            return 'cant load data'
        try:     
            df = filter_dataset(Id,data)
        except:
            return "cqnt filter"
        try:
            model = load_models()
        except:
            return "cant load model"
        prediction = model.predict_proba(df)[:, 1][0]
        return("Le client" + str(Id) + "a la probabilite suivante : " + str(prediction))
    except :
        return f"{Id} not found"

@app.route("/")
def hello():
    """
    Ping the API.
    """
    return jsonify({"Projet 7 Hamadi Prediction API is up et le modele est pret a etre utilise :)." : " Pour tester l'api vous pouvez utiliser le lien https://hamadicreditapp.herokuapp.com//test_api/?Id=xxxxxxxx" })


@app.route('/predict', methods=['POST'])
def predict():
    # parse input features from request
    request_json = request.get_json()
    df = pd.json_normalize(request_json)
 
    # load model
    model = load_models()
    prediction = model.predict_proba(df)[:, 1][0]
    # Format prediction in percentage with 2 decimal points
    prediction = "The client has a " + str(round(prediction*100,2)) + "% risk of defaulting on their loan."
    print("prediction: ", prediction)

    # Return output
    return jsonify(json.dumps(str(prediction)))

if __name__ == '__main__':
    app.run(debug=True)