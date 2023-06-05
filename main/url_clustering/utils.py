import json

import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import torch
from nltk.corpus import stopwords
import pymorphy2
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import umap.umap_ as umap
import os
from rutermextract import TermExtractor


logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_url_clustering_config(config_path):
    with open(config_path, 'r') as f:
        file = f.read()
        config = json.loads(file)
        return config


class Pages:
    def __init__(self):
        self.raw_data = None
        self.target_to_idx = None
        self.idx_to_target = None
        self.data = None

    @classmethod
    def load(cls, file_path):
        pages = Pages()
        try:
            with open(file_path, 'r') as f:
                file = f.read()
                data = json.loads(file)
        except json.decoder.JSONDecodeError:
            with open(file_path, 'r') as f:
                data = list()
                for line in f:
                    data.append(json.loads(line))
        pages.raw_data = data
        pages.form_dataset()
        return pages

    def form_dataset(self):
        targets = set()
        url_data = []
        for i in self.raw_data:
            targets.add(i['target'])

            target_data = {
                'target': i['target'],
                'url': i['url'],
                'url_text': i['url_text']}
            url_data.append(target_data)

        targets = sorted(list(targets))
        self.target_to_idx = {t: i for i, t in enumerate(targets)}
        self.idx_to_target = {i: t for i, t in enumerate(targets)}

        self.data = {idx: {} for idx in self.idx_to_target.keys()}
        for i in url_data:
            self.data[self.target_to_idx[i['target']]][i['url']] = i['url_text']

    def target_urls_text(self, target):
        idx = self.target_to_idx[target]
        return idx, self.data[idx]

    def add_topics(self, labeled_urls, target, topics):
        for idx, i in enumerate(self.raw_data):
            if i['target'] == target:
                topic = int(labeled_urls[i['url']])
                self.raw_data[idx]['topic'] = topic
                self.raw_data[idx]['topic_top_word'] = topics[topic]['top_n_words']

    def save(self, file_path):
        with open(f'{file_path}/pages_labeled.json', 'w') as f:
            json.dump(self.raw_data, f)

        with open(f'{file_path}/target_to_idx.json', 'w') as f:
            json.dump(self.target_to_idx, f)


def sentences_embeddings(sentences, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    t = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    embeddings = embeddings.cpu().numpy()

    return embeddings


def reduce_embedding_dimension(embeddings, new_emb_dim, random_state):
    umap_embeddings = umap.UMAP(n_components=new_emb_dim,
                                random_state=random_state)\
                        .fit_transform(embeddings)
    return umap_embeddings


def preprocess(text):
    stopword = stopwords.words('russian')
    morph = pymorphy2.MorphAnalyzer()

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation + '«' + '»')).strip()

    words = text.split()
    words = [word for word in words if
             word not in stopword and len(word) > 1]
    words = [morph.parse(word)[0].normal_form for word in words]

    return ' '.join(words)


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words="english").fit(
        documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(count, docs_per_topic, tf_idf, n=3):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [words[j] for j in indices[i]][::-1] for i, label
        in enumerate(labels)}

    return top_n_words


def term_c_tf_idf(doc, terms_idf):
    term_extractor = TermExtractor()
    terms = term_extractor(doc)
    terms_count = [term.count for term in terms]
    all_terms_count = sum(terms_count)
    tf = np.divide(terms_count, all_terms_count)
    terms_tf_idf = {term.normalized: terms_idf.get(term.normalized, 1.0) * i
                    for term, i in zip(terms, tf)}
    return terms_tf_idf


def labeled_dataset(embeddings, labels):
    labels = labels.reshape((len(labels), 1))
    dataset = np.concatenate((embeddings, labels), axis=1)
    cols = [f'emb{i}' for i in range(embeddings.shape[1])]
    cols.append('label')

    df = pd.DataFrame(dataset, columns=cols)
    return df


def mkdir(directory_name, directory_path=None):
    if directory_path:
        path = f'{directory_path}/{directory_name}'
    else:
        path = directory_name
    if not os.path.exists(path):
        os.makedirs(path)
    return path
