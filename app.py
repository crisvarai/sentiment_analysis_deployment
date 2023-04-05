import re
import string
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, url_for, request

def remove_pattern(in_text, pattern):
    r = re.findall(pattern, in_text)
    for i in r:
        in_text = re.sub(i, '', in_text)
    return in_text

def punct_count(in_text):
    count = sum([1 for char in in_text if char in string.punctuation])
    return round(count/(len(in_text) - in_text.count(" ")), 3)*100

app = Flask(__name__)

data = pd.read_csv("./dataset/sentiment.tsv", sep="\t")
data.columns = ['label', 'body_text']

# Data Preparation
data['label'] = data['label'].map({'pos': 0, 'neg':1})
vec_rp = np.vectorize(remove_pattern)
data['tidy_text'] = vec_rp(data['body_text'], '@[\w]*')
data['tidy_text'] = data['tidy_text'].str.replace('[^a-zA-z#]', ' ', regex=True)
tokenized = data['tidy_text'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized = tokenized.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized)):
    tokenized[i] = ' '.join(tokenized[i])
data['tidy_text'] = tokenized
data['body_length'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['%punctuation'] = data['body_text'].apply(lambda x: punct_count(x))

# Feature Selection
X, y = data['tidy_text'], data['label']
bow_vect = CountVectorizer(stop_words='english')
X = bow_vect.fit_transform(X)
X = pd.concat([data['body_length'], data['%punctuation'], pd.DataFrame(X.toarray())], axis=1)

# Classifier
clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, 
                         l1_ratio=None, max_iter=3000, n_jobs=None, penalty='l2', random_state=None, 
                         tol=0.0001, verbose=0, warm_start=False)
clf.fit(X, y)

# Flask App
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        msg = request.form['message']
        data = [msg]
        vect = pd.DataFrame(bow_vect.transform(data).toarray())
        body_lenght = pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([punct_count(data)])
        all_data = pd.concat([body_lenght, punct, vect], axis=1)
        prediction = clf.predict(all_data)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
