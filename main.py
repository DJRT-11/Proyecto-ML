import joblib, re, unicodedata, sklearn, nltk
import pandas as pd

from flask import Flask, render_template,request
from models.ScriptPreprocessing import TextPreprocessing_TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
import __main__
__main__.TextPreprocessing_TFIDF = TextPreprocessing_TFIDF

# Load models and steps
loaded_cls = joblib.load('models/pipeline_model.joblib')
loaded_clt_ppl = loaded_cls['pipeline']
loaded_clt_mdl = loaded_cls['model']
loaded_clt_tfi = loaded_cls['tfidf']
loaded_clt_svd = loaded_cls['svd']

loaded_reg = joblib.load('models/pipeline_model_Reg.joblib')
loaded_reg_ohj = loaded_reg['One_hot_job']
loaded_reg_ohx = loaded_reg['One_hot_exp']
loaded_reg_ohl = loaded_reg['One_hot_lab']
loaded_reg_mdl = loaded_reg['model']

# Predefined roles per label
roles_dict = {
    'label': [0,0,0,0,0,1,1,2,2,2,2,2],
    'job_title' : ['Data Engineer', 'Data Scientist', 'Machine Learning Engineer', 'Research Scientist', 'Data Science', 
              'Analytics Engineer', 'Director of Data Science', 
              'Data Analyst', 'Data Architect', 'Research Engineer', 'Research Analyst','Data Manager'],
    'experience_level' : ['','','','','','','','','','','',''],
    'remote_ratio' : ['','','','','','','','','','','','']
}
roles_df = pd.DataFrame(roles_dict)

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
        year = int(request.form['years'])
        remo = request.form['exampleRadios']

        data_procs = loaded_clt_ppl.transform(skil)
        data_tfidf = loaded_clt_tfi.transform(data_procs)
        data_svd = loaded_clt_svd.transform(data_tfidf)
        data_pred = loaded_clt_mdl.predict(data_svd)

        exp_lev = ''
        if 0 <= year < 2:
            exp_lev = 'en'
        elif 2 <= year < 5:
            exp_lev = 'mi'
        elif 5 <= year < 11:
            exp_lev = 'se'
        else:
            exp_lev = 'ex'
        
        roles_pred = roles_df[roles_df['label'] == data_pred[0]]
        roles_pred['label'] = 'label_'+roles_pred['label'].astype(str)
        roles_pred['experience_level'] = exp_lev
        roles_pred['remote_ratio'] = float(remo)

        salar_preds = []

        for i, rd in roles_pred.iterrows():
            row_df = pd.DataFrame([rd]).reset_index(drop=True)

            role_enc = pd.concat([
                row_df[['remote_ratio']],
                pd.DataFrame(loaded_reg_ohj.transform(row_df[['job_title']]), columns=loaded_reg_ohj.categories_[0]),
                pd.DataFrame(loaded_reg_ohx.transform(row_df[['experience_level']]), columns=loaded_reg_ohx.categories_[0]),
                pd.DataFrame(loaded_reg_ohl.transform(row_df[['label']]), columns=loaded_reg_ohl.categories_[0])
            ], axis=1)

            role_reg = {
                'job_title' : row_df.loc[0,'job_title'],
                'salary': round(loaded_reg_mdl.predict(role_enc)[0] / 500) * 500
            }

            salar_preds.append(role_reg)

        print(salar_preds)
        return render_template('prediction.html', name=name, mail=mail, skil=skil, year=year, cluster=data_pred[0], salaries=salar_preds)

@app.route('/test',methods=['GET'])
def test():
    jobs = [
        {
            'job_title': 'Engineer',
            'salary': 1000,
        },
        {
            'job_title': 'Data Scientist',
            'salary': 6000,
        },
        {
            'job_title': 'Manager',
            'salary': 4000,
        }
    ]
    return render_template('prediction.html', name="Brayan", mail="mail", skil="skil", year="3", cluster=2, salaries=jobs)


# Main

if __name__ == "__main__":
    app.run(debug=True)