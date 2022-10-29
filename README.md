### Классификатор мужских и женских голосов на базе [LibriTTS](https://arxiv.org/abs/1904.02882)

##### Создание окружения  

`conda create --name env_name --file requirements.txt`

#####  Загрузка параметров обученных моделей

Папку `parameters` с обученными моделями можно скачать с гугл [диска](https://drive.google.com/drive/folders/1foVXpHJjmczpn_8D1wfWS69t6KJ5jy_b?usp=sharing)

##### Запуск обученного классификатора на тестовой аудиозаписи

`python vc-inference.py --path=./test_inference/26_495_000004_000000.wav`

Отчет о работе в файле `Отчет.pdf`

