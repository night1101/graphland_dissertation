# graphland_dissertation
# Adaptive Information Aggregation in GNNs for Fraud Detection

Магистерская диссертация: разработка и исследование адаптивных методов агрегации
информации в графовых нейронных сетях для обнаружения мошеннических объектов.

## Текущий этап

Первый этап (по НТЗ, п.16): литературный обзор GraphLand, аудит данных,
EDA, воспроизведение базовых моделей на `tolokers-2` в режимах RL/RH/TH.

## Данные

Бенчмарк **GraphLand** (Bazhenov, Platonov, Prokhorenkova, NeurIPS 2025).
Основной датасет — `tolokers-2` (бинарная классификация вершин,
11 758 вершин, 1 038 000 направленных записей рёбер в PyG, 19 признаков
после кодирования категориальных полей).

Загрузка через `torch_geometric.datasets.GraphLandDataset`:

```python
from torch_geometric.datasets import GraphLandDataset
dataset = GraphLandDataset(root='data', name='tolokers-2', split='RH')
data = dataset[0]
```

Режимы разбиения: `RL` (10/10/80, случайное), `RH` (50/25/25, случайное),
`TH` (50/25/25, временное), `THI` (индуктивный, три снимка графа).

Полные данные в репозиторий не коммитятся. Инструкция по получению —
в `data/README.md` (Zenodo / Kaggle / автозагрузка через PyG).

## Структура репозитория

```
.
├── data/                      # сюда PyG скачивает GraphLand (не в git)
├── notebooks/
│   └── eda.ipynb              # EDA: аудит, баланс классов, homophily, KS-тест, THI
├── src/
│   ├── data_load/
│   │   └── load.py            # load_split / load_thi
│   ├── models/
│   │   ├── mlp.py             # MLP-канал (ego)
│   │   └── gcn.py             # GCN-канал (graph)
│   ├── evaluate.py            # AP, ROC-AUC, summary, paired t-test
│   └── train2.py              # основной скрипт baseline-экспериментов
├── experiments/
│   └── baseline/
│       ├── baseline_results_raw.csv
│       └── baseline_results_summary.csv
├── requirements.txt
└── README.md
```

## Воспроизведение baseline

```bash
pip install -r requirements.txt
python src/train2.py
```

Скрипт обучает MLP и GCN в режимах RL/RH/TH, по 5 seed (0–4),
сохраняет `baseline_results_raw.csv` и `baseline_results_summary.csv`
в `experiments/baseline/`.

### Протокол эксперимента

- Признаки стандартизуются `StandardScaler`, **fit только на train**
  каждого сплита — утечки нет.
- Гиперпараметры и число эпох фиксированы (`CONFIG` в `train2.py`):
  lr=1e-2, weight_decay=5e-4, hidden=64, epochs=300, patience=50.
- Чекпоинт выбирается по **val AP** (строгий порог улучшения 1e-6).
- **Test используется один раз** — отчётная оценка на выбранном по val чекпоинте.
  Test не участвует в выборе гиперпараметров, признаков или ранней остановки.
- `torch.set_num_threads(1)` — для воспроизводимости времени обучения.
- Один seed = один прогон; разброс по 5 seed — это вариация обучения,
  а не вариация по независимым графам (вершины одного графа зависимы).

### Метрики

Основная — **Average Precision (AP)** по предсказанным вероятностям,
как в исходной статье GraphLand. Дополнительно — ROC-AUC.
Для TH val/test prevalence сильно ниже, чем в train (~8% vs ~36%),
поэтому AP между RH и TH сравнивать напрямую нельзя без оговорки
(см. EDA, ячейка «интерпретация сравнимости режимов»).

## EDA (`notebooks/eda.ipynb`)

Покрывает:
- аудит признаков (dtype, NaN/Inf, изолированные вершины);
- баланс классов по режимам и сплитам;
- пересечения масок RL/RH/TH и вывод о сравнимости test-наборов;
- edge/node homophily на train-подграфе;
- распределение степеней;
- KS-тест сдвига признаков RH train vs TH test;
- загрузку THI и **проверку индуктивного сценария**:
  val/test маски указывают только на вершины, отсутствовавшие
  в предыдущем снимке (утечки нет).

## Известные ограничения (честно)

1. **THI**: данные загружаются и проверены на отсутствие утечки,
   но эксперименты на THI — второй этап (по НТЗ, E4).
2. **LightGBM-NFA** как сильный таблично-графовый baseline
   не реализован на первом этапе. По НТЗ (п.8) он обязателен
   ко второму этапу; в первом задании достаточно двух моделей
   (MLP + GCN), что и сделано.
3. **GAT** как сильный графовый baseline не добавлен на первом этапе;
   планируется во втором (п.16 п.4: «затем добавить GAT»).
4. **node_homophily в RL ≈ 0.05** — артефакт малого train-подграфа
   (1175 вершин), а не свойство графа. Для RL используем edge homophily
   как основную меру. Отмечено в EDA.
5. **TH val prevalence ≈ 8%** против train ≈ 36% — сильный сдвиг.
   Выбор чекпоинта по val AP в TH рискован, это отмечено в ограничениях;
   во втором этапе планируется анализ чувствительности к выбору
   критерия ранней остановки.
6. **Разброс по seed ≠ вариация по графам**. Вершины одного графа
   зависимы, поэтому p-value парного t-теста по 5 seed трактуется
   как описательная мера, а не как строгий вывод о популяции графов.

## Ссылки

- Статья GraphLand: https://arxiv.org/abs/2409.14500
- Репозиторий: https://github.com/yandex-research/graphland
- PyG-загрузчик: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GraphLand
- Данные: Zenodo (records/16895532), Kaggle