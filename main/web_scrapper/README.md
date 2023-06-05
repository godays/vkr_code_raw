configs: 
./web_scrapper/cfg/cfg.yml - web_scrapper config
./web_scrapper/cfg/logger_cfg.yml - python logger config

run:
python -m web_scrapper

Cейчас отключена возможность закачки данных через Wordstat, т.к. там требуется логин: __main__.py строка 102

Если падает WebDriver - скорее всего конфликт с текущей версией хрома.   
Скачать другую версию: https://chromedriver.chromium.org/downloads  
Положить в: web_scrapper/webdrivers/<os>/chromedriver  
