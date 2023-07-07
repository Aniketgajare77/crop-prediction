from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import sklearn
import os
import pickle
import warnings


app= Flask(__name__)

def predictiondata(input_data):
    se = pickle.load(open('scalingC.pkl', 'rb'))
    gau_reg = pickle.load(open('cropmodeling.pkl', 'rb'))
    X = se.transform(input_data)
    ans = gau_reg.predict(X)[0]
    return ans




@app.route('/')
def displayform():

    return render_template('home.html')


@app.route('/inputs' , methods=['POST'])
def getinput():
    N_SOIL = int(request.form['N_SOIL'])
    P_SOIL = int(request.form['P_SOIL'])
    K_SOIL = int(request.form['K_SOIL'])
    TEMPERATURE = float(request.form['TEMPERATURE'])
    HUMIDITY = float(request.form['HUMIDITY'])
    ph = float(request.form['ph'])
    RAINFALL = float(request.form['RAINFALL'])
    CROP_PRICE = float(request.form['CROP_PRICE'])

    input_data=pd.DataFrame(data=[[N_SOIL,P_SOIL,K_SOIL,TEMPERATURE,HUMIDITY,ph,RAINFALL,CROP_PRICE]],
                            columns=['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'CROP_PRICE'])
    
    prediction = predictiondata(input_data)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction in crop_dict:
        crop=crop_dict[prediction]
        result='{} is the best crop that you can grow here'.format(crop)
    else:
        result='sorry we can not recommend crop of provided data.'
    
    return render_template('display.html',data=result)









if __name__=='__main__':
    app.run(debug=True)