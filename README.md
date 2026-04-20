# AIConf Butterflies

Материалы мастер-класса AIConf 2026 года по сбору датасета и обучению модели. 

## Что нужно для запуска

- Python 3.12
- `requirements.txt`
- Jupyter Notebook / JupyterLab для работы с ноутбуками

Создание окружения и установка зависимостей:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
```

## Датасеты на Hugging Face

- Основной датасет, из которого черпаются все следующие: https://huggingface.co/datasets/vsevolod-nv/aiconf-butterfly-detection-all
- Расширенный goldenset - результат разметки: https://huggingface.co/datasets/vsevolod-nv/aiconf-butterfly-detection-goldenset-extended
- Замер качества qwen: https://huggingface.co/datasets/vsevolod-nv/aiconf-butterfly-qwen-eval
- Замер качества gpt-4o-mini:

## Структура проекта

### `1-scrape`

Сбор исходного датасета из iNaturalist, первичная фильтрация изображений и выгрузка датасета aiconf-butterfly-detection-all.

### `2-markup-learn`

Ноутбук для массовой разметки learn-датасета VLM-моделью без расчета метрик качества и выгрузки результата в датасет `aiconf-butterfly-learn`.

### `2-tuning`

Пробный ноутбук с обучением YOLO на маленьком goldenset и получением бейзлайна качества.

### `3-tasks`

Подготовка задач разметки: review, bbox, экзамен для разметчиков, майнинг большего goldenset и сборка расширенного эталонного набора.

### `4-vlm`

Пишем промпт и оцениваем качество разметки большими моделями: `qwen3-5-397b-a17b-fp8`, `gpt-4o-mini`, ``. Последующая разметка моделью целевого датасета для обучения YOLO.

### `5-tuning`

Обучение YOLO и получение итоговых метрик.
