import pandas as pd
import numpy as np
from threading import Thread
from flask import Flask,render_template,request,redirect
import simplejson as json
import requests
import datetime
import jinja2
import pickle

import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.utils import column_or_1d
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

app.pred={}
app.rec={}
# Load pipe_rf from saved predictive_model
pipe_rf = pickle.load(open('predictive_model.sav', 'rb'))
# Load df_rec from saved recommender_dataframe
df_rec = pd.read_pickle('recommender_dataframe')

# Compute jaccard similarity
def get_jaccard_sim(str1, str2): 
   a = set(str1.split()) 
   b = set(str2.split())
   c = a.intersection(b)
   return float(len(c)) / (len(a) + len(b) - len(c))

def get_recommendation(name, platform):
   df_rec1 = df_rec[['Name', 'Platform', 'Joined Column']]
   df_rec_final = df_rec1[df_rec1['Platform'] == platform]
   df_rec_final = df_rec_final[['Name', 'Joined Column']]
   game_desc = df_rec_final.groupby('Name')['Joined Column'].apply(list).to_dict() # Convert dataframe to dictionary
   rec_dict = {}
   for k in game_desc.keys():
      rec_dict[k] = get_jaccard_sim(game_desc[name][0], game_desc[k][0])
   sort_dict = sorted(rec_dict.items(), key=lambda x: x[1], reverse=True)
   return [tuple[0] for tuple in sort_dict[1:6]]

@app.route('/')   
def index():
   return render_template("index.html")

@app.route('/predictor', methods=['GET', 'POST'])
def predict():
   app.pred['platform'] = request.json['platform']
   app.pred['genre'] = request.json['genre']
   app.pred['publisher'] = request.json['publisher']
   app.pred['critic_score'] = request.json['critic_score']
   data = pd.DataFrame([(app.pred['platform'], app.pred['genre'], app.pred['publisher'],  app.pred['critic_score'])], columns = ['Platform' , 'Genre', 'Publisher' , 'Critic_Score'])
   return str(round(pipe_rf.predict(data).tolist()[0], 2))
 
@app.route('/recommender', methods=['GET', 'POST'])
def recommend():
   app.rec['platform_rec'] = request.json['platform_rec']
   app.rec['name'] = request.json['name']
   rec = get_recommendation(app.rec['name'].upper(), app.rec['platform_rec'])
   str_rec = "\n".join(rec)
   return str_rec

if __name__ == '__main__':
   app.run(port=33507)