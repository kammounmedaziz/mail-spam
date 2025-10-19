# ==============================
# 1️⃣ Import Libraries
# ==============================
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # <-- for saving/loading models

# Download stopwords
nltk.download('stopwords')


# ==============================
# 2️⃣ Load and Inspect Data
# ==============================
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
df = pd.read_csv(DATA_DIR / 'spam_ham_dataset.csv')
df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ' '))

print(df.info())
print(df.head())


# ==============================
# 3️⃣ Text Preprocessing
# ==============================
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

corpus = []
for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df['label_num']


# ==============================
# 4️⃣ Train Model
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

print("✅ Model trained successfully!")
print(f"📊 Accuracy on test data: {clf.score(X_test, y_test) * 100:.2f}%")


# ==============================
# 5️⃣ Save Model and Vectorizer
# ==============================
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(clf, MODELS_DIR / 'spam_detector_model.pkl')
joblib.dump(vectorizer, MODELS_DIR / 'vectorizer.pkl')
print(f"💾 Model and vectorizer saved successfully to: {MODELS_DIR}")


# ==============================
# 6️⃣ Test Example Prediction
# ==============================
email_to_classify = df['text'].iloc[10]

text = email_to_classify.lower()
text = text.translate(str.maketrans('', '', string.punctuation)).split()
text = [stemmer.stem(word) for word in text if word not in stopwords_set]
email_clean = ' '.join(text)

X_email = vectorizer.transform([email_clean]).toarray()
prediction = clf.predict(X_email)[0]

print("\n🔎 Predicted label:", "SPAM" if prediction == 1 else "HAM")
print("📌 True label:", df['label'].iloc[10])
