# Модуль для классификации ссылок

Используется берт.

Пример использования

**[Веса модели](https://drive.google.com/file/d/1uH46a0zRmGy3yg7fDAI8_hDLthmFUp-u/view?usp=share_link)**

```
import classifier
url_clf = classifier.UrlClassifier(weights='bert.pt')
url_clf.predict_for_text(TEXT) # если хотим классифицировать голый текст
url_clf.predict_for_url(URL) # парсит все теги <title> и запускает функцию выше
```
