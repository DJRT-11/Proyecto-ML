import joblib, re, unicodedata, sklearn, pandas, nltk
# import numpy as np

from flask import Flask, render_template,request
from models import TextPreprocessing_TFIDF
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

# nltk.download('stopwords')

app = Flask(__name__)

# Load model and steps
loaded_file = joblib.load('models\pipeline_model.joblib')
print(loaded_file)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_post',methods=['POST'])
def handle_post():
    if request.method == 'POST':
        name = request.form['user']
        mail = request.form['mail']
        skil = request.form['skills']
        year = request.form['years']

        skil = skil.replace(',',', ')

        return f'<h2>Nombre: {name}<h2> <h2>Correo: {mail}<h2> <h2>Habilidades: {skil}<h2> <h2>Años de exp.: {year}<h2>'



# Main

if __name__ == "__main__":
    app.run(debug=True)