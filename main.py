import joblib, re, unicodedata, sklearn, pandas, nltk
# import numpy as np

from flask import Flask, render_template,request
from models.ScriptPreprocessing import TextPreprocessing_TFIDF
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
import __main__
__main__.TextPreprocessing_TFIDF = TextPreprocessing_TFIDF

# Load model and steps
loaded_file = joblib.load('models/pipeline_model.joblib')
loaded_pipeline = loaded_file['pipeline']
loaded_model = loaded_file['model']
loaded_tfidf = loaded_file['tfidf']
loaded_svd = loaded_file['svd']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def handle_post():
    if request.method == 'POST':
        name = request.form['user']
        mail = request.form['mail']
        skil = request.form['skills']
        year = request.form['years']

        skil = skil.replace(',',', ')

        data_procs = loaded_pipeline.transform(skil)
        data_tfidf = loaded_tfidf.transform(data_procs)
        data_svd = loaded_svd.transform(data_tfidf)
        data_pred = loaded_model.predict(data_svd)

        return f'<h2>Nombre: {name}</h2> <h2>Correo: {mail}</h2> <h2>Habilidades: {skil}</h2> <h2>Años de exp.: {year}</h2> <h2>Cluster predicho: {data_pred}</h2>'



# Main

if __name__ == "__main__":
    app.run(debug=True)