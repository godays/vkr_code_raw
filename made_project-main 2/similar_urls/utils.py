import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import logging
import torch
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string
import os


logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_url_text(file_path):
    return pd.read_json(file_path, lines=True)


class TextPrepocess:
    def __init__(self, model_path, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        corpus = list(corpus)
        t = self.tokenizer(corpus, padding=True,
                           truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(
                **{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        embeddings = embeddings.cpu().numpy()

        self.encoded_corpus = embeddings

    def encode_text(self, text):
        t = self.tokenizer(
            self.preprocess_text(text), padding=True,
            truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(
                **{k: v.to(self.model.device) for k, v in t.items()})

        text_embedding = model_output.last_hidden_state[:, 0, :]
        text_embeddings = torch.nn.functional.normalize(text_embedding)
        text_embedding = text_embedding.cpu().numpy()

        return text_embedding

    def preprocess_text(self, text):
        punct = "".join(list(set([',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
                                  '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
                                  '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
                                  '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
                                  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√'] + list(string.punctuation))))

        text = re.sub(r'\:(.*?)\:', '', text)
        text = str(text).lower()  # Making Text Lowercase
        text = re.sub('\[.*?\]', '', text)
        # The next 2 lines remove html text
        text = BeautifulSoup(text, 'lxml').get_text()
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = text.translate(str.maketrans('', '', punct)).strip()
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
        text = re.sub(r"[^а-яА-Яa-zA-Z?.!,¿']+", " ", text)
        text = ' '.join([t.strip() for t in text.split() if t.strip()])

        return text

    def preprocess_texts(self, sentences):
        '''Cleaning and parsing the text.'''
        clean_sents = [self.preprocess_text(x) for x in sentences]

        return clean_sents
