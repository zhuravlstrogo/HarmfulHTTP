## 1 ML задача

Тебе даны данные HTTP запросов. 

Глобальная задача — найти или разделить вредоносные от хороших. Как мы знаем, вредоносных классов может быть несколько. 

Важнее отделить «мух от котлет». 

Попробуй сделать EDA, понять, а точно ли данные не избыточны и всё, что ты вообще можешь сказать?!

В ходе решения этой части ожидаем, что будут предоставлены артефакты (например jupyter notebook) с экспериментами, которые помогут понять, почему принято решение использовать такой подход к задаче.

<details>
  <summary>- Подсказка 1</summary>
    Классов может быть до 50, но не обязательно 50
</details>
    
<details>
  <summary>- Подсказка 2</summary>
    Кластеризация очень помогает
</details>
    
<details>
  <summary>- Подсказка 3</summary>
    Да, типично что у нас нет числа классов. Но что делать, это жизнь? Мы для базового решения использовали DBSCAN, может быть и ты можешь начать с него?
</details>

<details>
  <summary>- Подсказка 4</summary>
    посмотри на поля, точно ли все они нужны?
</details>

## 2 Сервис для inference

Наши задачи заканчивается тогда, когда решение внедрено в прод. 

Поэтому вторая часть посвящена inference части. 

Представь, что те запросы — это поток данных, которые приходят, или ты можешь их забирать из какого-то внешнего сервиса. 

Твоя задача — реализовать интерфейс сервис — принимать каким-либо образом запросы и отдавать ответы — номер класса. Ты можешь самостоятельно определить, что за что будет отвечать.

Можешь в тесты? - отлично, покажи что можешь. Можешь в Docker? - именно этого нам и надо! умеешь настраивать несложные CI — нам уже очень нравится!

У нас есть тестовые примеры, и мы бы хотели запустить их на твоем коде. 

<details>
  <summary>Нам для проверки решения удобно, когда формат общений с сервисом унифицирован. Можно предложить свой вариант.</summary>
  
  Но предлагаем посмотреть на [openapi.json](https://gist.github.com/amurzina/dac86af14de5dd100c1cda82cf442f9b) и реализовать метод  predict, например так:

    Запрос

  ```bash
    curl -X 'POST' \
    'http://127.0.0.1:80/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '[{"data": "{\"CLIENT_IP\": \"188.138.92.55\", \"CLIENT_USERAGENT\": NaN, \"REQUEST_SIZE\": 166, \"RESPONSE_CODE\": 404, \"MATCHED_VARIABLE_SRC\": \"REQUEST_URI\", \"MATCHED_VARIABLE_NAME\": NaN, \"MATCHED_VARIABLE_VALUE\": \"//tmp/20160925122692indo.php.vob\", \"EVENT_ID\": \"AVdhXFgVq1Ppo9zF5Fxu\"}"}, {"data": "{\"CLIENT_IP\": \"93.158.215.131\", \"CLIENT_USERAGENT\": \"Mozilla/5.0 (Windows NT 6.3; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0\", \"REQUEST_SIZE\": 431, \"RESPONSE_CODE\": 302, \"MATCHED_VARIABLE_SRC\": \"REQUEST_GET_ARGS\", \"MATCHED_VARIABLE_NAME\": \"url\", \"MATCHED_VARIABLE_VALUE\": \"http://www.galitsios.gr/?option=com_k2\", \"EVENT_ID\": \"AVdcJmIIq1Ppo9zF2YIp\"}"}]'
  ```
  Ответ от твоего сервиса:

```bash
[
    {
    "EVENT_ID": "AVdhXFgVq1Ppo9zF5Fxu",
    "LABEL_PRED": 42
    },
    {
    "EVENT_ID": "AVdcJmIIq1Ppo9zF2YIp",
    "LABEL_PRED": 3
    }
    ]
```

</details>

Если приведешь пример, как можно автоматически присылать такие ответы, взяв за образец выданный csv, будет здорово.

---

Решение мы ожидаем получить в виде доступа к приватному репозиторию. Для GitHub добавь пожалуйста: amurzina, nlyf, reviewer-pt. И сообщи пожалуйста Тане @tanyasmirom когда готово, чтобы точно не потеряли.

---