from flask import Flask,render_template,request,jsonify
from flask_socketio import SocketIO,emit

import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY','THE_KEY')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


socketio = SocketIO(app,cors_allowed_origins="*")

model_pipeline = None
feature_columns = None

@app.before_request
def load_model_on_startup():
    global model_pipeline,feature_columns
    if model_pipeline is None:
        try:
            print(f"Current Working Directory: {os.getcwd()}") 
            print(f"Path to app.py: {os.path.abspath(__file__)}")

            model_pipeline_path = os.path.join(BASE_DIR,'best_fraud_detection_model.joblib')
            feature_columns_path = os.path.join(BASE_DIR,'feature_columns.joblib')

            print(f"Looking for model at: {os.path.join(os.getcwd(), model_pipeline_path)}") 
            print(f"Looking for features at: {os.path.join(os.getcwd(), feature_columns_path)}")


            if not os.path.exists(model_pipeline_path):
                print("Model File Not found")
                return 
            if not os.path.exists(feature_columns_path):
                print('Feature Columns Path not found')
                return
            
            model_pipeline = joblib.load(model_pipeline_path)
            feature_columns = joblib.load(feature_columns_path)
        except Exception as e:
            print(f"error found in loading Model or Features")
            model_pipeline=None
            feature_columns=None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('new_transaction')
def handle_new_transaction(json_data):
    if model_pipeline is None or feature_columns is None:
        emit('prediction_result' , {"error" : 'Server error : Model Not Loaded.'}) 
        return
    try:
        input_features = {col: json_data.get(col,0.0) for col in feature_columns}

        input_df = pd.DataFrame([input_features],columns=feature_columns)      

        #Make Predictions
        predictions = model_pipeline.predict(input_df)[0]
        prediction_proba = model_pipeline.predict_proba(input_df)[:,1][0]

        result = "Fraud" if predictions == 1 else "Not Fraud"
        result_color = "red" if predictions == 1 else "green"    

        emit('prediction_result',{
            'prediction' : result,
            'fraud_probability' : float(prediction_proba),
            'color' : result_color
        })    
    except Exception as e:
        print(f"Prediction Error {e}")


if __name__ == '__main__':
    socketio.run(app,debug=True,host='0.0.0.0',port=5000,allow_unsafe_werkzeug=True)

