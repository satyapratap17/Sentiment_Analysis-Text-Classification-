from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder  # Import the LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

app = Flask(__name__)

# Load the dataset
education = pd.read_csv('Education.csv')
politics = pd.read_csv('Politics.csv')
sports = pd.read_csv('Sports.csv')
finance = pd.read_csv('Finance.csv')

df = pd.concat([education, sports, finance, politics], ignore_index=True)

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
cv = CountVectorizer(max_features=2000)
lb = LabelEncoder()  # Define the LabelEncoder


def preprocessingText(text):
    text = re.sub("[^a-zA-Z0-9]", "", text)
    text = text.lower()
    return text


def preprocessingTextUsingNLTK(text):
    text = preprocessingText(text)
    allEngStopWords = stopwords.words("english")
    if text != None:
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split(" ") if word not in set(allEngStopWords)])
    else:
        print(text)
    return text


df['Text'] = df['Text'].apply(preprocessingTextUsingNLTK)
y = lb.fit_transform(df['Label'])
df.drop(columns=['Label'], inplace=True)
arr = cv.fit_transform(df['Text']).toarray()
X = pd.DataFrame(arr)

y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.2)
gnb = GaussianNB()
gnb.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        text = preprocessingTextUsingNLTK(text)
        text_vectorized = cv.transform([text]).toarray()
        prediction = gnb.predict(text_vectorized)
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

