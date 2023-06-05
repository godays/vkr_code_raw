import os
from flask import Flask
from flask import request
from flask import render_template
from knn_classifier import KNNClassifier


app = Flask(__name__, template_folder='./templates', static_folder='./templates/static')
#url_clf = UrlClassifier(weights='/home/model/bert.pt')
url_clf = KNNClassifier(weights="./model/tfidf_model.pkl")


@app.route('/', methods=['GET'])
def text():
    text = request.args.get('text')
    url = request.args.get('url')
    app.logger.warn("Processing text1 {}".format(os.listdir(".")))
    predict = None
    if text is not None:
        app.logger.warn("Processing text1 {}".format(text))
        predict = url_clf.predict_for_text(text)[0]
        app.logger.warn("For text {} - predict {}".format(text, predict))
    elif url is not None:
        app.logger.warn("Processing url {}".format(url))
        predict = url_clf.predict_for_url(url)[0]
        app.logger.warn("For url {} - predict {}".format(url, predict))

    return render_template('main.html', predict=predict)

if __name__ == '__main__':
    app.run()
