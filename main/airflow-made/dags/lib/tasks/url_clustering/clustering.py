import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import json
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lib.tasks.url_clustering import utils
import umap
import pickle


class URLClustering:
    def __init__(self, embeddings, reduce_emb_dim=False, new_emb_dim=5,
                 random_state=None, save_artefacts=False, artefacts_path=None):
        self.reduce_emb_dim = reduce_emb_dim
        self.new_emb_dim = new_emb_dim
        self.random_state = random_state
        self.save_artefacts = save_artefacts
        self.artefacts_path = artefacts_path

        if self.reduce_emb_dim:
            self.embeddings = utils.reduce_embedding_dimension(embeddings,
                                                          self.new_emb_dim,
                                                          self.random_state)
        else:
            self.embeddings = embeddings

        self.model = None
        self.labels_ = None

    @staticmethod
    def plot_dendrogram(model, **kwargs):
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        dendrogram(linkage_matrix, **kwargs)

    def plot_url_dendrogram(self, target=None, truncate_mode=None,
                            p=3):
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model = model.fit(self.embeddings)

        plt.figure(figsize=(12, 7))
        self.plot_dendrogram(model, color_threshold=0,
                             truncate_mode=truncate_mode, p=p)
        plt.xlabel(
            "Number of points in node (or index of point if no parenthesis)")
        target_title = f' (target = {target})' if target else ''
        plt.title(f"Hierarchical Clustering Dendrogram{target_title}")

        if self.save_artefacts:
            plt.savefig(f'{self.artefacts_path}/dendrogram.png',
                        bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def fit(self, n_clusters=2):
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
        self.model.fit(self.embeddings)

        self.labels_ = self.model.labels_


class TopicExtractor:
    def __init__(self, url_sentences, labels, top_n_words=3,
                 random_state=None, save_artefacts=False, artefacts_path=None):
        self.random_state = random_state
        self.save_artefacts = save_artefacts
        self.artefacts_path = artefacts_path
        self.docs_df, self.docs_per_topic = self.preprocess_url_sentences(
            url_sentences, labels)

        self.tf_idf, self.count = utils.c_tf_idf(
            self.docs_per_topic.Doc.values,
            m=self.docs_df.shape[0])

        self.top_n_words = self.extract_top_n_words_per_topic(top_n_words)
        self.add_topic_to_docs_df()

    @staticmethod
    def preprocess_url_sentences(url_sentences, labels):
        sentences = list(url_sentences.values())
        norm_sentences = [utils.preprocess(s) for s in sentences]

        docs_df = pd.DataFrame({"url": url_sentences.keys(),
                                "Doc": norm_sentences})
        docs_df['topic'] = labels
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['topic'], as_index=False).agg(
            {'Doc': ' '.join})

        return docs_df, docs_per_topic

    def extract_top_n_words_per_topic(self, n=3):
        words = self.count.get_feature_names_out()
        labels = list(self.docs_per_topic.topic)
        tf_idf_transposed = self.tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]

        top_n_words = self.extract_topic_sizes()
        for i, label in enumerate(labels):
            top_n_words[label]['top_n_words'] = [words[j] for j in indices[i]][::-1]

        if self.save_artefacts:
            with open(f'{self.artefacts_path}/topics.json', 'w') as f:
                json.dump(top_n_words, f)

        return top_n_words

    def extract_topic_sizes(self):
        topic_sizes = (self.docs_df.groupby(['topic'])
                       .Doc
                       .count()
                       .reset_index()
                       .rename({"Doc": "Size"}, axis='columns')
                       .set_index('topic')
                       .to_dict('index'))
        return topic_sizes

    def add_topic_to_docs_df(self):
        self.docs_df['topic_top_word'] = self.docs_df['topic'].map(
            lambda x: self.top_n_words[x]['top_n_words'])

    def print_topics(self):
        df = pd.DataFrame(self.top_n_words).T
        df.reset_index(drop=False, inplace=True)
        df['top_n_words'] = df['top_n_words'].map(lambda x: ' '.join(x))
        max_str = np.max(df['top_n_words'].str.len())
        df['top_n_words'] = df['top_n_words'].str.ljust(max_str)
        print(df.to_string(index=False, header=['topic', 'size', '']))

    def plot_topics(self, embeddings, target=None):
        embeddings = utils.reduce_embedding_dimension(embeddings,
                                                new_emb_dim=2,
                                                random_state=self.random_state
                                                )

        plot_df = pd.DataFrame(embeddings, columns=['x', 'y'])
        plot_df['labels'] = self.docs_df.topic
        plot_df['topic_top_word'] = self.docs_df.apply(
            lambda row: f"{row['topic']} - {' '.join(row['topic_top_word'])}",
            axis=1)
        plot_df.sort_values('labels', inplace=True)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=plot_df, x='x', y='y', hue='topic_top_word')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.xlabel('umap embedding 0')
        plt.ylabel('umap embedding 1')
        target_title = f'target = {target}' if target else ''
        plt.title(target_title)

        if self.save_artefacts:
            plt.savefig(f'{self.artefacts_path}/clusters_scatterplot.png',
                        bbox_inches='tight')

        plt.tight_layout()
        plt.show()


class TopicClassifier:
    def __init__(self, data=None, random_state=None, save_artefacts=False,
                 artefacts_path=None):
        self.random_state = random_state
        self.save_artefacts = save_artefacts
        self.artefacts_path = artefacts_path
        self.model = KNeighborsClassifier()
        self.data = data
        self.train = None
        self.test = None
        self.umap_model = umap.UMAP(n_components=5,
                                    random_state=self.random_state)
        self.umap_plot = umap.UMAP(n_components=2,
                                   random_state=self.random_state)
        self.train_embeddings = None
        self.test_embeddings = None
        self.test_prediction = None

    def fit(self):
        self.train, self.test = train_test_split(self.data, test_size=0.1,
                                                 stratify=self.data['label'],
                                                 random_state=self.random_state
                                                 )
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

        self.train_embeddings = self.umap_model.fit_transform(self.train.drop(
            'label', axis=1))

        self.test_embeddings = self.umap_model.transform(self.test.drop(
            'label', axis=1))

        y = self.train['label']
        self.model.fit(self.train_embeddings, y)

    def predict(self, embeddings):
        embeddings = self.umap_model.transform(embeddings)
        return self.model.predict(embeddings)

    def print_test_metrics(self):
        y = self.test['label']
        self.test_prediction = self.model.predict(self.test_embeddings)
        pred_prob = self.model.predict_proba(self.test_embeddings)
        roc_auc = roc_auc_score(y, pred_prob, multi_class='ovo',
                                average='macro')
        print(f'Test sample ROC AUC = {roc_auc:.2f}')

    def plot_topics(self, target=None, top_n_words=None):
        train_embeddings = self.umap_plot.fit_transform(self.train.drop('label',
                                                                    axis=1))
        test_embeddings = self.umap_plot.transform(self.test.drop('label', axis=1))

        train_plot_df = pd.DataFrame(train_embeddings, columns=['x', 'y'])
        train_plot_df['true_label'] = self.train['label']
        train_plot_df['label'] = self.train['label']
        train_plot_df['split'] = 'train'

        test_plot_df = pd.DataFrame(test_embeddings, columns=['x', 'y'])
        test_plot_df['true_label'] = self.test['label']
        test_plot_df['label'] = self.test_prediction
        test_plot_df['split'] = 'test'

        plot_df = pd.concat([train_plot_df, test_plot_df])
        plot_df.sort_values(['true_label', 'split'], inplace=True)
        plot_df.reset_index(drop=True, inplace=True)

        cmap = mpl.cm.get_cmap('icefire')
        color_normalizer = mpl.colors.Normalize()
        labels_norm = color_normalizer(plot_df['true_label'])
        labels_colors = cmap(labels_norm)

        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(data=plot_df, x='x', y='y', hue='label',
                        palette='icefire', edgecolor=labels_colors,
                        style='split', linewidth=1.5,
                             legend=True)

        if top_n_words:
            for t in ax.get_legend().texts:
                if t.get_text() not in ('label', 'split', 'train', 'test'):
                    label = int(float(t.get_text()))
                    t.set_text(f'{label} - {" ".join(top_n_words[label]["top_n_words"])}')

        ax.get_legend().set_bbox_to_anchor(bbox=(1.02, 1))
        plt.xlabel('umap embedding 0')
        plt.ylabel('umap embedding 1')
        target_title = f'target = {target}' if target else ''
        plt.title(f'{target_title}\nЦвет круга - прогноз классификатора. Цвет границы - метка кластеризации.')

        if self.save_artefacts:
            plt.savefig(f'{self.artefacts_path}/classification_scatterplot.png',
                        bbox_inches='tight')

            self._plot_for_train(target=target, top_n_words=top_n_words)

        plt.tight_layout()
        plt.show()

    def _plot_for_train(self, target=None, top_n_words=None):
        train_embeddings = self.umap_plot.transform(self.train.drop('label', axis=1))

        train_plot_df = pd.DataFrame(train_embeddings, columns=['x', 'y'])
        train_plot_df['label'] = self.train['label']
        train_plot_df.sort_values(['label'], inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        p = sns.scatterplot(data=train_plot_df, x='x', y='y', hue='label',
                             palette='icefire', legend=True, ax=ax)

        if top_n_words:
            for t in p.get_legend().texts:
                if t.get_text() not in ('label', 'split', 'train', 'test'):
                    label = int(float(t.get_text()))
                    t.set_text(
                        f'{label} - {" ".join(top_n_words[label]["top_n_words"])}')

        p.get_legend().set_bbox_to_anchor(bbox=(1.02, 1))
        ax.set_xlabel('umap embedding 0')
        ax.set_ylabel('umap embedding 1')
        target_title = f'target = {target}' if target else ''
        ax.set_title(target_title)

        with open(f'{self.artefacts_path}/classification_plot_template.pkl', 'wb') as f:
            pickle.dump(ax, f)

        plt.close()

    def save(self):
        with open(f'{self.artefacts_path}/umap.pkl', 'wb') as f:
            pickle.dump(self.umap_model, f)

        with open(f'{self.artefacts_path}/classifier.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        with open(f'{self.artefacts_path}/umap_plot.pkl', 'wb') as f:
            pickle.dump(self.umap_plot, f)

    @classmethod
    def load(cls, artefacts_path):
        clf = TopicClassifier()

        with open(f'{artefacts_path}/umap.pkl', 'rb') as f:
            umap_model = pickle.load(f)

        with open(f'{artefacts_path}/classifier.pkl', 'rb') as f:
            model = pickle.load(f)

        with open(f'{artefacts_path}/umap_plot.pkl', 'rb') as f:
            umap_plot = pickle.load(f)

        clf.artefacts_path = artefacts_path
        clf.umap_model = umap_model
        clf.model = model
        clf.umap_plot = umap_plot
        return clf

    def plot_new_url_topic(self, embeddings, label, save_path):
        with open(f'{self.artefacts_path}/classification_plot_template.pkl', 'rb') as f:
            ax = pickle.load(f)

        test_embeddings = self.umap_plot.transform(embeddings)
        plot_df = pd.DataFrame(test_embeddings, columns=['x', 'y'])
        plot_df['label'] = [label]

        plt.scatter(plot_df['x'], plot_df['y'], color='r', marker='X')
        plt.text(plot_df['x'][0], plot_df['y'], int(label[0]))
        plt.tight_layout()

        plt.savefig(f'{save_path}/new_url.png', bbox_inches='tight')
        plt.close()
