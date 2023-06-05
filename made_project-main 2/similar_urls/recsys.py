from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle
import requests
from bs4 import BeautifulSoup

import utils

URL_FILE_PATH = 'pages.json'
BERT_MODEL_PATH = 'cointegrated/rubert-tiny'
SAVE_PATH = 'knn_model'
TOP_K = 10


class RecSys:
    def __init__(self, url_data=URL_FILE_PATH, pretrained=True, save_path=SAVE_PATH):
        self.save_path = save_path
        self.df = utils.load_url_text(file_path=url_data)

        self.tp = utils.TextPrepocess(BERT_MODEL_PATH, self.df.url_text)
        self.urls = self.df.url

        if pretrained:
            self.model = pickle.load(open(self.save_path, 'rb'))
        else:
            self.embeddings = self.tp.encoded_corpus

    def train(self):
        X = self.embeddings

        knn = KNeighborsClassifier(
            metric='cosine', n_neighbors=13, weights='distance')

        knn.fit(X, self.urls)

        self.model = knn

        knnPickle = open(self.save_path, 'wb')
        pickle.dump(knn, knnPickle)
        knnPickle.close()

    def get_text_from_url(self, url):
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        description_tag = soup.find('meta', {'name': 'description'})

        if description_tag is not None:
            description = description_tag.get('content')
        else:
            description = ''

        titles = []
        for title in soup.find_all('title'):
            titles.append(title.get_text())

        return ' '.join(titles + [description])

    def get_top_urls(self, url):
        url_text = self.get_text_from_url(url)
        clean_text = self.tp.preprocess_text(url_text)

        emb_text = self.tp.encode_text(clean_text)

        most_similar_urls = self.model.kneighbors(
            emb_text, n_neighbors=TOP_K)[1][0]

        return [self.urls[i] for i in most_similar_urls]
