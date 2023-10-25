#modules import
from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np  
#main code
app = Flask(__name__,template_folder='template')
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open("RidgeModel.pkl",'rb'))
@app.route('/')
def index():
    location = sorted(data['location'].unique())
    return render_template('index.html',locations=location)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')
    print(location,bhk,bath,sqft)
    input = pd.DataFrame([[location,bhk,bath,sqft]],columns=['location','bhk','bath','sqft'])
    prediction = pipe.predict(input)[0]*1e5
    return str(np.round(prediction,2))

if __name__=="__main__":
    app.run(debug=True,port=5000)
