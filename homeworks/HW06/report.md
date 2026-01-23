## 1. Dataset

Для исследования был выбран датасет S06-hw-dataset-01.csv. Это задача бинарной классификации с умеренным дисбалансом классов — распределение составляет примерно 67% / 33%.

Ключевые характеристики данных:

- **Тип задачи:** Бинарная классификация
- **Баланс классов:** Умеренный дисбаланс (67%/33%)
- **Признаки:** 24 числовых (num_), 6 категориальных (cat_)
- **Общий размер:** ~30 признаков, 10000+ наблюдений

Данные подходят для проверки деревьев решений и ансамблей из-за смешанного типа признаков.

## 2. Protocol

Предобработка и протокол эксперимента:

- **Обработка пропусков:** числовые — медиана, категориальные — мода
- **Кодирование категориальных:** One-Hot Encoding
- **Масштабирование:** `StandardScaler` для моделей, чувствительных к масштабу
- **Разделение:** `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` (80/20, стратификация)
- **Кросс-валидация:** GridSearchCV, `cv=5`, оптимизация по `roc_auc` (приоритет)

Baseline:

Dummy Classifier (strategy='most_frequent') — метрики: Accuracy ~0.676, F1 0.0, ROC-AUC 0.5

Logistic Regression (Pipeline: `StandardScaler` + `LogisticRegression`) — Accuracy ~0.827, F1 ~0.707, ROC-AUC ~0.874

## 3. Models

Рассматривались и настраивались следующие алгоритмы:

- `DecisionTree` (GridSearch по сложности: `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`, `ccp_alpha`)
- `RandomForest` (осн. параметры: `n_estimators`, `max_depth`, `min_samples_leaf`)
- `Boosting` (HistGradientBoosting — подбор `learning_rate`, `max_iter`, `max_depth`)

Для каждого алгоритма выполнен GridSearchCV и сохранены лучшие параметры в `artifacts/search_summaries.json`.

## 4. Results

Сравнение по тестовой выборке (извлечено в `artifacts/metrics_test.json`):

- Dummy: Accuracy 0.677, F1 0.000, ROC-AUC 0.500
- LogisticRegression: Accuracy 0.828, F1 0.708, ROC-AUC 0.875
- DecisionTree: Accuracy 0.877, F1 0.800, ROC-AUC 0.907
- RandomForest: Accuracy 0.929, F1 0.885, ROC-AUC 0.967
- Boosting: Accuracy 0.939, F1 0.903, ROC-AUC 0.974

Лучшие параметры (сохранены в `artifacts/search_summaries.json`):

Boosting (победитель):

```
{
    'learning_rate': 0.2,
    'max_depth': 10,
    'max_iter': 200
}
```

RandomForest:

```
{
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_leaf': 1
}
```

DecisionTree:

```
{
    'max_depth': 10,
    'min_samples_leaf': 10,
    'min_samples_split': 2,
    'max_leaf_nodes': null,
    'ccp_alpha': 0.0
}
```

## 5. Analysis

Permutation importance (метрика: падение ROC-AUC) показала топ-признаки:

- `num18` (важность ~0.060)
- `num19` (важность ~0.045)
- `num07` (важность ~0.031)
- `num04` (важность ~0.017)
- `num01` (важность ~0.013)

Наблюдения:

- Числовые признаки доминируют в топе важности.
- Категориальные признаки занимают средние позиции.
- Признаки `num18` и `num19` несут сильный сигнал для классификации.

## 6. Conclusion

Рекомендуемая модель для продакшна: `Boosting` (HistGradientBoosting) с оптимальными гиперпараметрами, сохранённая модель и метаданные находятся в `artifacts/`.

Дальнейшие шаги: мониторинг метрик, периодический ретренинг, объяснимость (SHAP) и эксперименты с балансировкой классов.