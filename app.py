import pandas as pd
import numpy as np
from threading import Thread
import io

from flask import Flask,render_template,request,redirect, Response
import simplejson as json
import requests
import datetime
import jinja2
import pickle
import os
import base64
import random

import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.utils import column_or_1d
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, Legend
from bokeh.palettes import Category20
from bokeh.plotting import figure
from bokeh.models import CustomJS, Dropdown, Select
from bokeh.embed import components

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from wordcloud import WordCloud
import seaborn as sns

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# set file directory path
MODEL_PATH = os.path.join(APP_ROOT, 'predictive_model.sav')  
# set path to the model
pipe_rf = pickle.load(open(MODEL_PATH, 'rb')) 
# load the pickled model

# set file directory path
DATA_PATH = os.path.join(APP_ROOT, 'recommender_dataframe')  
# set path to the data
df_rec = pickle.load(open(DATA_PATH, 'rb')) 
# load the pickled data

# set file directory path
VIZ_PATH = os.path.join(APP_ROOT, 'visual_dataframe')  
# set path to the viz
df_model = pickle.load(open(VIZ_PATH, 'rb')) 
# load the pickled viz

app.pred={}
app.rec={}

# Compute jaccard similarity
def get_jaccard_sim(str1, str2): 
   a = set(str1.split()) 
   b = set(str2.split())
   c = a.intersection(b)
   return float(len(c)) / (len(a) + len(b) - len(c))

# def get_recommendation(name, platform):
#    df_rec1 = df_rec[['Name', 'Platform', 'Joined Column']]
#    df_rec_final = df_rec1[df_rec1['Platform'] == platform]
#    df_rec_final = df_rec_final[['Name', 'Joined Column']]
#    game_desc = df_rec_final.groupby('Name')['Joined Column'].apply(list).to_dict() # Convert dataframe to dictionary
#    rec_dict = {}
#    for k in game_desc.keys():
#       rec_dict[k] = get_jaccard_sim(game_desc[name][0], game_desc[k][0])
#    sort_dict = sorted(rec_dict.items(), key=lambda x: x[1], reverse=True)
#    return [tp[0] + "," + " sim_score: " + str(round(tp[1],2)) for tp in sort_dict[1:11]]

def get_recommendation(name, platform):
   df_rec1 = df_rec[['Name', 'Platform', 'Joined Column']]
   df_rec_final = df_rec1[df_rec1['Platform'] == platform]
   df_rec_final = df_rec_final[['Name', 'Joined Column']]
   game_desc = df_rec_final.groupby('Name')['Joined Column'].apply(list).to_dict() # Convert dataframe to dictionary
   rec_dict = {}
   for k in game_desc.keys():
      rec_dict[k] = get_jaccard_sim(game_desc[name][0], game_desc[k][0])
   sort_dict = sorted(rec_dict.items(), key=lambda x: x[1], reverse=True)
   return {tp[0]: round(tp[1],2) for tp in sort_dict[1:11]}


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
 
# @app.route('/recommender', methods=['GET', 'POST'])
# def recommend():
#    app.rec['platform_rec'] = request.json['platform_rec']
#    app.rec['name'] = request.json['name']
#    rec = get_recommendation(app.rec['name'].upper(), app.rec['platform_rec'])
#    str_rec = "\n".join(rec)
#    return str_rec

@app.route('/recommender', methods=['GET', 'POST'])
def recommend():
   app.rec['platform_rec'] = request.json['platform_rec']
   app.rec['name'] = request.json['name']

   print(request.json['platform_rec'])
   print(request.json['name'])

   rec = get_recommendation(app.rec['name'].upper(), app.rec['platform_rec'])

   game_list = list(rec.keys())
   percent = [int(x*100) for x in list(rec.values())]

   # number of data points
   n = len(percent)
   # percent of circle to draw for the largest circle
   percent_circle = max(percent) / 100

   r = 1  # outer radius of the chart
   r_inner = 0.1  # inner radius of the chart
   # calculate width of each ring
   w = (r - r_inner) / n

   # create colors along a chosen colormap
   colors = [plt.cm.magma(i / n) for i in range(n)]
   # colors.reverse()
   # colors = plt.cm.tab10.colors

   # Circular barplot in python with percentage labels
   fig = Figure(figsize=(12,8))
   ax = fig.add_subplot(1, 1, 1)
   ax.axis("equal")

   for i in range(n):
      radius = r - i * w
      ax.pie([percent[i] / max(percent) * percent_circle], radius=radius, startangle=90, normalize=False,
            counterclock=False,
            colors=[colors[i]],
            labels=[f'{game_list[i]} – {percent[i]}%'], labeldistance=None,
            wedgeprops={'width': w, 'edgecolor': 'white'})
      ax.text(0, radius - w / 2, f'{game_list[i]} – {percent[i]}% ', ha='right', va='center')

   canvas = FigureCanvas(fig)
   output = io.BytesIO()
   canvas.print_png(output)
   response = Response(output.getvalue())
   response.mimetype = 'image/png'

   return response

# @app.route('/plot.png', methods=['GET', 'POST'])
# def plot():
   
#    fig = Figure()
#    axis = fig.add_subplot(1, 1, 1)

#    xs = range(100)
#    ys = [random.randint(1, 50) for x in xs]

#    axis.plot(xs, ys)
#    canvas = FigureCanvas(fig)
#    output = io.BytesIO()
#    canvas.print_png(output)
#    response = Response(output.getvalue())
#    response.mimetype = 'image/png'
#    return response


# if __name__ == '__main__':
#    app.run(port=33507)