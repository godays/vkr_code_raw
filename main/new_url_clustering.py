from url_clustering.clustering import URLClustering, TopicExtractor, TopicClassifier
from url_clustering import utils
import json


BERT_MODEL_DIRECTORY_PATH = "rubert-tiny"
TARGET_TO_IDX_PATH = "data/output/target_to_idx.json"
CLASSIFIER_PATH = "data/output/clustering"

URL = 'https://www.lockobank.ru/personal/karty/kreditnye/'
URL_TEXT = 'Оформить кредитную карту в Локо-Банк | Подать заявку онлайн'
TARGET = 'кредитная карта'
SAVE_PATH = '.'

with open(TARGET_TO_IDX_PATH, 'r') as f:
    target_to_idx = json.load(f)

target_idx = target_to_idx[TARGET]
clustering_path = f'{CLASSIFIER_PATH}/target_{target_idx}'

with open(f'{clustering_path}/topics.json', 'r') as f:
    top_n_words = json.load(f)

url_sentences = {URL: URL_TEXT}
sentences = list(url_sentences.values())
embeddings = utils.sentences_embeddings(sentences=sentences,
                                                model_path=BERT_MODEL_DIRECTORY_PATH)

clf = TopicClassifier.load(clustering_path)
label = clf.predict(embeddings)
label_top_word = top_n_words[str(int(label))]['top_n_words']
clf.plot_new_url_topic(embeddings, label, save_path=SAVE_PATH)
