import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import numpy as np
import urllib3.request as urllib
from flask import (Flask, redirect, render_template, request, send_file, Response, url_for)
import os
from io import StringIO

import re
import string

from flask import (Flask, redirect, render_template, request,
                   send_from_directory,Response, url_for)

app = Flask(__name__)

Lab = {0: 'Bank Charges and Fees', 1: 'Groceries', 2: 'Transport and Fuel', 3: 'Cellphone',
       4: 'Restaurants and Take-Aways', 5: 'Entertainment', 6: 'Internet and Telephone', 7: 'Holidays and Travel',
       8: 'Clothing', 9:'Gambling'}

list_10cat = ['Bank Charges and Fees', 'Groceries', 'Transport and Fuel', 'Cellphone', 'Restaurants and Take-Aways', 
              'Entertainment', 'Internet and Telephone', 'Holidays and Travel', 'Clothing', 'Gambling']

# @app.route('/model', methods=['GET'])
# def download():
#     # pth=os.getcwd()
#     try:
#         # model = os.path.join(pth, 'static/models/TFModel02.h5')
#         model = "https://mldevworspace8102427333.blob.core.windows.net/model-container/TFModel02.h5"
#         return send_file(model, as_attachment=True)
#     except Exception as e:
#         return (str(e))

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
   result = ""
   if file:
       #TODO: Classify Content and return Result Content as CSV FIle
       df = pd.read_csv(file)  # file.file

       df['Description_New'] = df['Description'].apply(lambda x: semi_clean(x))
       df["Combo"] = np.where((df["Reference"].notnull()) & (df["Description"] != df["Reference"]) & (df["Reference"].str.isnumeric() == False), df["Description"] + ' ' + df["Reference"], df["Description"])
       df.drop(columns=['Description','Reference'], axis=1, inplace = True)
       df.rename(columns={'Combo':'Description'}, inplace=True)
       df = df.loc[df["CategoryDescription"].isin(list_10cat)]
       data = df["Description_New"]
       
       try:
           preprocessor = hub.KerasLayer("universal-sentence-encoder-cmlm_multilingual-preprocess_2")
           #req = urllib.urlopen("https://mldevworspace8102427333.blob.core.windows.net/model-container/TFModel02.h5")
           req = urllib.urlopen("https://mldevworspace8102427333.blob.core.windows.net/model-container/TFModel02.h5")
           mdl = req.read()

           LoadModel_TF02 = tf.keras.models.load_model(mdl, compile=False, custom_objects={'KerasLayer':preprocessor})

           pred_val = get_categories(data)
           #print(pred_val)
           
           Pred_Data = pd.concat([df,pred_val], axis = 1)
           Pred_Data = Pred_Data.drop(Pred_Data.columns[0],axis=1)
           #print(Pred_Data)
           s = StringIO()
           Pred_Data.to_csv(s, index=False)

           print('Request for classification received as file name=%s' % file.filename)
           result = s.getvalue()
       except Exception as e: # work on python 3.x
            result = str(e)
       return Response(result, mimetype='text/csv')
  
       
   else:
       print('Request for classification received without file -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()
