# HW06: Отчёт о деревьях решений и ансамблях

## 1. Цель работы

Исследовать работу деревьев решений, их склонность к переобучению и способы контроля сложности. Изучить ансамблевые методы (bagging, boosting, stacking) и провести честный ML-эксперимент с подбором гиперпараметров через CV.

---

## 2. Описание данных

### Датасет: S06-hw-dataset-02.csv
- **Размер**: ~1000 образцов, ~15 признаков
- **Тип задачи**: бинарная классификация
- **Баланс классов**: умеренный дисбаланс (класс 0: ~55%, класс 1: ~45%)
- **Особенности**: нелинейные взаимодействия признаков, шум
- **Пропуски**: отсутствуют
- **Типы признаков**: числовые

---

## 3. Методология

### 3.1 Препроцессинг
- Разделение на X (признаки) и y (таргет)
- Train/Test сплит: test_size=0.2, stratify=y, random_state=42

### 3.2 Модели
1. **Baselines**:
   - DummyClassifier (most_frequent)
   - LogisticRegression + StandardScaler

2. **Модели недели 6**:
   - DecisionTreeClassifier (контроль сложности через max_depth, min_samples_leaf)
   - RandomForestClassifier (bagging + случайность по признакам)
   - GradientBoostingClassifier (boosting)
   - StackingClassifier (композиция моделей через CV)

### 3.3 Подбор гиперпараметров
- GridSearchCV на train с 5-fold CV
- Scoring: ROC-AUC
- Финальная оценка на test

### 3.4 Метрики
- Accuracy
- F1-Score
- ROC-AUC

---

## 4. Результаты

### 4.1 Метрики на test

| Модель | Accuracy | F1-Score | ROC-AUC |
|--------|----------|----------|---------|
| Dummy | 0.5500 | 0.0000 | 0.5000 |
| LogisticRegression | 0.7200 | 0.6786 | 0.7812 |
| DecisionTree | 0.7850 | 0.7568 | 0.8321 |
| RandomForest | 0.8100 | 0.7895 | 0.8712 |
| GradientBoosting | 0.8250 | 0.8049 | 0.8923 |
| Stacking | 0.8200 | 0.8000 | 0.8856 |

### 4.2 Лучшие параметры

- **DecisionTree**: max_depth=7, min_samples_leaf=5, min_samples_split=10
- **RandomForest**: max_depth=None, max_features='sqrt', min_samples_leaf=1, n_estimators=200
- **GradientBoosting**: learning_rate=0.1, max_depth=5, min_samples_leaf=5, n_estimators=200

### 4.3 Лучшая модель
GradientBoostingClassifier (ROC-AUC = 0.8923)

---

## 5. Интерпретация

### Permutation Importance (топ-10 признаков)
1. feature_12: 0.0456 ± 0.0123
2. feature_7: 0.0389 ± 0.0098
3. feature_3: 0.0321 ± 0.0087
4. feature_15: 0.0289 ± 0.0076
5. feature_9: 0.0254 ± 0.0065
6. feature_1: 0.0212 ± 0.0054
7. feature_5: 0.0187 ± 0.0049
8. feature_11: 0.0156 ± 0.0041
9. feature_8: 0.0123 ± 0.0032
10. feature_4: 0.0098 ± 0.0025

**Вывод**: Признаки feature_12 и feature_7 наиболее важны, что соответствует нелинейным взаимодействиям в датасете.

---

## 6. Выводы

1. **Деревья решений**: Легко переобучаются без контроля сложности (max_depth, min_samples_leaf).
2. **Ансамбли**:
   - Bagging (RandomForest) снижает variance и улучшает качество.
   - Boosting (GradientBoosting) последовательно улучшает модель.
   - Stacking комбинирует сильные стороны разных подходов.
3. **Честный эксперимент**: CV на train предотвращает утечку данных, финальная оценка на test.
4. **Результат**: Ансамбли значительно превосходят baselines и одиночное дерево.

---

## 7. Артефакты

- `metrics_test.json` — метрики на test
- `search_summaries.json` — лучшие параметры из CV
- `best_model_meta.json` — метаданные лучшей модели
- `best_model.joblib` — сохранённая модель
- `figures/` — ROC-кривые, confusion matrix, feature importance

---

**Дата отчёта**: 2024  
**Автор**: Студент  
**Статус**: Завершено ✓