# Email Spam Detection

A simple email spam vs ham classifier using scikit-learn. It trains a RandomForest model on the classic spam/ham dataset and saves the model artifacts in the `models/` folder.

## Project Structure

```
.
├── data/
│   └── spam_ham_dataset.csv
├── models/
├── src/
│   ├── pipeline.py
│   └── train.py
└── README.md
```

## Setup

1. Create and activate a virtual environment (recommended).
2. Install required packages:
   - pandas, numpy, nltk, scikit-learn, joblib
3. Download NLTK stopwords (handled in code on first run).

## Train

From the project root:

```
python .\src\train.py
```

This will:
- load `data/spam_ham_dataset.csv`
- preprocess and vectorize text
- train a RandomForest classifier
- save artifacts to `models/spam_detector_model.pkl` and `models/vectorizer.pkl`

## Notes

- Model artifacts are ignored by Git via `.gitignore`. If you need to version them, remove the patterns in `.gitignore`.
- Ensure the dataset exists at `data/spam_ham_dataset.csv`.

## Next Steps

- Add a prediction script (e.g., `src/predict.py`) that loads artifacts from `models/` and classifies new text.
- Add `requirements.txt` to pin dependencies for reproducible installs.
