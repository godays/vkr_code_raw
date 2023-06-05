import requests
from bs4 import BeautifulSoup

from bert_dataset import CustomDataset
from bert_classifier import BertClassifier

decoder = {0: 'банкротство', 1: 'казино', 2: 'кредитная карта', 3: 'мотоцикл'}


class UrlClassifier:
    def __init__(self, weights='bert.pt'):
        self.model = BertClassifier(model_path='cointegrated/rubert-tiny', tokenizer_path='cointegrated/rubert-tiny',
                                    n_classes=4, epochs=15, model_save_path='bert.pt', inference_path=weights)

    def predict_for_text(self, text):
        preds = decoder[self.model.predict(text)]
        return preds

    def predict_for_url(self, url):
        text = self.get_text_from_url(url)
        preds = self.predict_for_text(text)
        return preds

    def get_text_from_url(self, url):
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        titles = []
        for title in soup.find_all('title'):
            titles.append(title.get_text())

        return ' '.join(titles)
