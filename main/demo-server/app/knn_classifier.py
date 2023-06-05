import os
import pickle
import re
import string
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, n_neighbors=13, save_state=False, weights=None):
        self.n_neighbors = n_neighbors
        self.save_state = save_state
        self.tfidf = None
        self.model = None
        self.classes_ = None
        if weights is not None:
            self.load(weights)

    def fit(self, X, y):
        X = [self.preprocess_text(t) for t in X]
        self.tfidf = TfidfVectorizer()
        X = self.tfidf.fit_transform(X)
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric='cosine')
        self.model.fit(X, y)
        self.classes_ = np.unique(y)

        if self.save_state:
            self.save()

    def get_text_from_url(self, url):
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        titles = []
        for title in soup.find_all('title'):
            titles.append(title.get_text())

        return ' '.join(titles)

    def predict(self, X):
        X = self.preprocess_text(X)
        X = self.tfidf.transform([X])
        return self.model.predict(X)

    def predict_for_text(self, X):
        return self.predict(X)

    def predict_for_url(self, url):
        text = self.get_text_from_url(url)
        preds = self.predict_for_text(text)
        return preds

    def save(self, filepath='tfidf_model.pkl'):
        if not self.tfidf or not self.model:
            raise ValueError(
                'Model and Tf-Idf representation must be fitted before saving the state.')

        state = {
            'tfidf': self.tfidf,
            'model': self.model,
            'classes_': self.classes_
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath='tfidf_model.pkl'):
        if not os.path.exists(filepath):
            raise ValueError(f'Invalid filepath: {filepath}')

        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.tfidf = state['tfidf']
        self.model = state['model']
        self.classes_ = state['classes_']

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra white space
        text = re.sub(r'\s+', ' ', text)

        # Remove leading and trailing white space
        text = text.strip()

        return text