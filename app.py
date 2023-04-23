import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np

import os
from io import StringIO

import re
import string

from flask import (Flask, redirect, render_template, request,
                   send_from_directory,Response, url_for)

app = Flask(__name__)

preprocessor = hub.KerasLayer("universal-sentence-encoder-cmlm_multilingual-preprocess_2")
LoadModel_TF02 = tf.keras.models.load_model('TFModel02.h5', compile=False, custom_objects={'KerasLayer':preprocessor})

Lab = {0: 'Bank Charges and Fees', 1: 'Groceries', 2: 'Transport and Fuel', 3: 'Cellphone',
       4: 'Restaurants and Take-Aways', 5: 'Entertainment', 6: 'Internet and Telephone', 7: 'Holidays and Travel',
       8: 'Clothing', 9:'Gambling'}

list_10cat = ['Bank Charges and Fees', 'Groceries', 'Transport and Fuel', 'Cellphone', 'Restaurants and Take-Aways', 
              'Entertainment', 'Internet and Telephone', 'Holidays and Travel', 'Clothing', 'Gambling']

def semi_clean(text):
    final_string = ""
    text = text.lower()
    text = re.sub(r'\n', '', text)
    translator = str.maketrans('.', ' ')
    text = text.translate(translator)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    final_string = text
    return final_string
def predict(model, data):
    pred =  model.predict(data)
    return list(map(lambda x: (Lab[x.argmax()], x.max()), pred))
def get_categories(data):
    pred_data = pd.DataFrame(np.array(predict(LoadModel_TF02, data)))
    pred_data.columns = ["Prediction Category", "Prediction Probability"]
    return pred_data

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/classifyresult', methods=['POST'])
def classify():
   file = request.files['file']

   if file:

       #TODO: Classify Content and return Result Content as CSV FIle
       df = pd.read_csv(file)  # file.file

    
       df['Description_New'] = df['Description'].apply(lambda x: semi_clean(x))
       df["Combo"] = np.where((df["Reference"].notnull()) & (df["Description"] != df["Reference"]) & (df["Reference"].str.isnumeric() == False), df["Description"] + ' ' + df["Reference"], df["Description"])
       df.drop(columns=['Description','Reference'], axis=1, inplace = True)
       df.rename(columns={'Combo':'Description'}, inplace=True)
       df = df.loc[df["CategoryDescription"].isin(list_10cat)]
       
       data = df["Description_New"]
       pred_val = get_categories(data)
       #print(pred_val)
       
       Pred_Data = pd.concat([df,pred_val], axis = 1)
       Pred_Data = Pred_Data.drop(Pred_Data.columns[0],axis=1)
       #print(Pred_Data)
       s = StringIO()
       Pred_Data.to_csv(s, index=False)

       print('Request for classification received as file name=%s' % file.filename)
       return Response(s.getvalue(),mimetype='text/csv')
   else:
       print('Request for classification received without file -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()
