# Обзор

Тестовое задание ML в PT [описание](https://github.com/zhuravlstrogo/harmful_http/blob/main/task.md)

## Структура проекта

### `/data`

исходный датасет и промежуточные файлы (скрыты в .gitignore)

### `/notebooks`

0.1.feature_transform.ipynb - основной ноутбук исследования и преобразования фичей
0.2.alternative_feature_transform.ipynb - альтернативный вариант преобразования категориальных фичей (не доработан)
1.0.select_num_clusters.ipynb - графики подбора оптимального количества кластеров
2.1.make_clusters.ipynb - основной ноутбук с различными алгоритмами кластеризации
2.2.pycaret.ipynb - дополнительный ноутбук с различными алгоритмами кластеризации (не доработан)

### `/src/make_clusters`
preprocessiing.py make_clusters.py - скрипты для инференса на данных, которые лежат в data/ с названием 'part_10.csv.'. Результаты сохраняются в data/result.csv.

log.py - для сохранения логов в src/make_cluster/logs/update.logs

environment.yml - зависимости проекта, для установки из корня проекта:

```bash
conda env create -f /src/make_cluster/environment.yml
```

Для того, чтобы запустить в докер контейнере:

```bash 
docker run make-cluster:1.0.0
```
