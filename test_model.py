# ==============================
#  🧠 Spam Detection - Test Script
# ==============================

import string
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords (only first time)
nltk.download('stopwords')

# ==============================
# 1️⃣ Load Saved Model and Vectorizer
# ==============================
clf = joblib.load('../models/spam_detector_model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')
print("✅ Model and vectorizer loaded successfully!")

# ==============================
# 2️⃣ Preprocessing Function
# ==============================
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess_email(text):
    """Clean, tokenize, remove stopwords, and stem the email text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

# ==============================
# 3️⃣ Classify Any Email
# ==============================
# You can replace this string with any email text
email_input = """
Subject: Congratulations! You've won a $1000 Walmart gift card.
Click here to claim your prize now before it expires tomorrow!
"""

# Preprocess the email
clean_email = preprocess_email(email_input)

# Transform using the trained vectorizer
X_email = vectorizer.transform([clean_email]).toarray()

# Predict
prediction = clf.predict(X_email)[0]

# ==============================
# 4️⃣ Show Result
# ==============================
if prediction == 1:
    print("\n🚨 This email is **SPAM**.")
else:
    print("\n📩 This email is **HAM** (not spam).")
