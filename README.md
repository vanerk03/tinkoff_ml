# Утилита, которая на основе заданных текстов генерирует новые.
## Разработана для Английского языка

- Использует Двуграммную модель, на её основе строит список новых слов и делает выбор с помощью Sklearn Bagging модели (model.pkl), обущенной на Word2vec, выдающей вероятность следующего слова по векторам слова и двух предыдущих.

- Модель строит N-граммы только на заданых при инициализации текстах из них выбирает слова/контекст для генерации

- При передаче префикса длина сгенерированного текста будет включать длину префикса: вы передали префикс длины 5 и хотите сгенеривовать последовательность длины 10, в таком случае будет сгенерировано 5 новых слов.

-**Слова из префикса для генерации должны присутствовать в текстах из `input-dir` хотя бы один раз**

## Примеры сгенерированной последовательности:
### Параметры
```
prefix = "I love"
length = 10
input-dir: a few english Novels
```
### Вывод
`I love it when a girl's really a pretty childish.`
