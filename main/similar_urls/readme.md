# Модуль для выдачи ссылок

По умолчанию параметр **TOP_K** = 10.
Пример использования для обученной модели

```
import recsys

rs = recsys.RecSys()
rs.get_top_urls(URL) - > TOP_K ссылок
```

Пример если нужно заново обучить

```
rs = recsys.RecSys(url_data=URL_FILE_PATH, pretrained=False)
rs.fit()
```
