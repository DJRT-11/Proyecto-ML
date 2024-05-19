import joblib, re, unicodedata, sklearn, pandas, nltk

from flask import Flask, render_template,request
from models import TextPreprocessing_TFIDF


app = Flask(__name__)

# Load model and steps
# loaded_file = joblib.load('models\pipeline.joblib')
# print(loaded_file.steps)

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



        print(request.form)
        return f'<h2>Nombre: {name}<h2> <h2>Correo: {mail}<h2> <h2>Habilidades: {skil}<h2> <h2>Años de exp.: {year}<h2>'



# Main

if __name__ == "__main__":
    app.run(debug=True)