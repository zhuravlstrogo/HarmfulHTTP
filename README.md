# Обзор

Тестовое задание ML в PT [описание](https://github.com/zhuravlstrogo/harmful_http/blob/main/task.md)

## Структура проекта

### `/data`

исходный датасет и промежуточные файлы (скрыты в .gitignore)

### `/notebooks`

0.1.feature_transform.ipynb - основной ноутбук исследования и преобразования фичей :white_check_mark:

0.2.alternative_feature_transform.ipynb - альтернативный вариант преобразования категориальных фичей (в разработке :hourglass:)

1.0.select_num_clusters.ipynb - графики подбора оптимального количества кластеров :white_check_mark:

2.1.make_clusters.ipynb - основной ноутбук с различными алгоритмами кластеризации :white_check_mark:

2.2.pycaret.ipynb - дополнительный ноутбук с различными алгоритмами кластеризации (в разработке :hourglass:)

### `/src/make_clusters`
preprocessiing.py dbscan.py - скрипты для инференса на данных, которые лежат в data/part_10.csv (название захардкожено). Результаты сохраняются в data/result.csv.

log.py - для сохранения логов в src/make_cluster/logs/update.logs

environment.yml - зависимости проекта, для установки из корня проекта:

```bash
conda env create -f /src/make_cluster/environment.yml
```

Для того, чтобы запустить инференс в докер контейнере:

```bash 
docker run make-cluster:1.0.0
```
