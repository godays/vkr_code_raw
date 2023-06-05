from lib.tasks.url_clustering.clustering import URLClustering, TopicExtractor, TopicClassifier
from lib.tasks.url_clustering import utils

CFG = {
  "URLClustering": {
    "init": {
      "reduce_emb_dim": True,
      "new_emb_dim": 5,
      "random_state": 0,
      "save_artefacts": True
    },
    "plot_url_dendrogram": {
      "truncate_mode": "level",
      "p": 6
    }
  },
  "TopicExtractor": {
    "init": {
      "top_n_words": 3,
      "random_state": 0,
      "save_artefacts": True
    }
  },
  "TopicClassifier": {
      "init": {
        "random_state": 0,
        "save_artefacts": True
      }
  }
}


def url_clustering_main(
    page_content_path,
    data_output,
    bert_model_path,
    data_output_directory,
    save_artifacts
):

    config = CFG

    config['PAGES_FILE_PATH'] = page_content_path
    config['PAGES_OUTPUT_PATH'] = data_output
    config['BERT_MODEL_DIRECTORY_PATH'] = bert_model_path
    config['OUTPUT_DIRECTORY_PATH'] = data_output_directory
    config['SAVE_ARTIFACTS'] = save_artifacts

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
