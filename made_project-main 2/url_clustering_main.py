import argparse

from url_clustering.clustering import URLClustering, TopicExtractor, TopicClassifier
from url_clustering import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path')
    args = parser.parse_args()

    config = utils.load_url_clustering_config(args.config_path)

    PAGES_FILE_PATH = config['PAGES_FILE_PATH']
    PAGES_OUTPUT_PATH = config['PAGES_OUTPUT_PATH']
    BERT_MODEL_DIRECTORY_PATH = config['BERT_MODEL_DIRECTORY_PATH']
    OUTPUT_DIRECTORY_PATH = config['OUTPUT_DIRECTORY_PATH']

    pages = utils.Pages.load(PAGES_FILE_PATH)

    for target in pages.target_to_idx.keys():
        target_idx, url_sentences = pages.target_urls_text(target)

        target_path = utils.mkdir(f'target_{target_idx}',
                                  OUTPUT_DIRECTORY_PATH)

        sentences = list(url_sentences.values())
        embeddings = utils.sentences_embeddings(sentences=sentences,
                                                model_path=BERT_MODEL_DIRECTORY_PATH)

        cluster_model = URLClustering(embeddings, artefacts_path=target_path,
                                      **config['URLClustering']['init'])
        cluster_model.plot_url_dendrogram(target=target,
                                          **config['URLClustering'][
                                              'plot_url_dendrogram'])

        print('Enter number of clusters:')
        n_clusters = int(input())

        cluster_model.fit(n_clusters=n_clusters)

        topic_extr = TopicExtractor(url_sentences=url_sentences,
                                    labels=cluster_model.labels_,
                                    artefacts_path=target_path,
                                    **config['TopicExtractor']['init'])

        topic_extr.print_topics()

        topic_extr.plot_topics(embeddings, target=target)

        labeled_urls = dict(zip(url_sentences.keys(), cluster_model.labels_))
        pages.add_topics(labeled_urls, target, topic_extr.top_n_words)

        df = utils.labeled_dataset(embeddings, cluster_model.labels_)
        clf = TopicClassifier(df, artefacts_path=target_path,
                              **config['TopicClassifier']['init'])
        clf.fit()
        clf.print_test_metrics()
        clf.plot_topics(target=target, top_n_words=topic_extr.top_n_words)
        clf.save()

    pages.save(PAGES_OUTPUT_PATH)
