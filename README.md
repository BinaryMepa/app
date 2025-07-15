# Dota2Enjoyer App

Приложение на Python/Jupyter для сегментации изображений (возможно, связанных с Dota 2).

## ⚙️ Возможности

- Обученная модель сегментации (`segmentation_model.pth`)
- Jupyter-ноутбук для демонстрации: `Untitled1.ipynb`
- Тестовый скрипт для воспроизведения результатов: `test.py`

## 📁 Содержимое репозитория

| Файл | Описание |
|---|---|
| `segmentation_model.pth` | Предобученная модель сегментации |
| `Untitled1.ipynb` | Jupyter-ноутбук с примерами использования модели |
| `test.py` | Скрипт для тестирования модели на новых изображениях |

## 🛠 Установка зависимости

Рекомендуется использовать виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate      # на Windows: venv\Scripts\activate
pip install -r requirements.txt
