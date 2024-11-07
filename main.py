import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib
import re
from translate import Translator  

app = Flask(__name__)

def clean_text(text):
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def translate_text(text, to_lang="en"):
    translator = Translator(to_lang=to_lang)
    translated_text = translator.translate(text)
    return translated_text

def load_data():
    file_path = "train.csv"
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl')
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx.")
    
    df.columns = df.columns.str.strip()
    
    print(df)
    print("processing begin >>>")

    if 'crimeaditionalinfo' not in df.columns:
        raise KeyError("The required column 'crimeaditionalinfo' is missing from the data.")
    
    df['crimeaditionalinfo'] = df['crimeaditionalinfo'].astype(str)
    df['crimeaditionalinfo'] = df['crimeaditionalinfo'].apply(clean_text)

    return df

def train_model():
    df = load_data()
    df['category'] = df['category'].fillna('')
    df['sub_category'] = df['sub_category'].fillna('')

    X = translate_text(df['crimeaditionalinfo'])
    y_category = df['category']
    category_model = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(LogisticRegression()))
    category_model.fit(X, y_category)
    joblib.dump(category_model, 'category_model.joblib')

    sub_category_models = {}
    unique_categories = df['category'].unique()

    for category in unique_categories:
        category_specific_df = df[df['category'] == category]
        if len(category_specific_df['sub_category'].unique()) > 1:  
            X_sub = category_specific_df['crimeaditionalinfo']
            y_sub = category_specific_df['sub_category']
            sub_model = make_pipeline(TfidfVectorizer(), OneVsRestClassifier(LogisticRegression()))
            sub_model.fit(X_sub, y_sub)
            sub_category_models[category] = sub_model 

    joblib.dump(sub_category_models, 'sub_category_models.joblib')
    print("Models trained successfully.")

def predict(message):
    processed_message = message
    
    category_model = joblib.load('category_model.joblib')
    sub_category_models = joblib.load('sub_category_models.joblib')

    category_prediction = category_model.predict([processed_message])
    category = category_prediction[0] if category_prediction else None

    sub_category = None
    if category in sub_category_models:
        sub_category_model = sub_category_models[category]
        sub_category_prediction = sub_category_model.predict([processed_message])
        sub_category = sub_category_prediction[0] if sub_category_prediction else None

    return category, sub_category

@app.route('/train', methods=['GET'])
def train():
    train_model()
    return "Model trained successfully."

@app.route('/predict', methods=['POST', 'GET'])
def predict_function():
    if request.method == 'POST':
        message = translate_text(clean_text(request.form['message']))
        category, sub_category = predict(message)  
        return render_template('predict.html', category=category, sub_category=sub_category, message=message)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
