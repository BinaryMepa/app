"""Streamlit demo: semantic segmentation of satellite images with a custom U-Net.

Run with:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image

from src.classes import CLASSES
from src.config import MODEL_PATH, SUPPORTED_IMAGE_TYPES
from src.inference import load_model, predict_mask
from src.visualization import colorize_mask

TITLE = "Модель для сегментации спутниковых снимков"

INTRO = """
Здравствуйте! Данное приложение было создано с целью демонстрации дипломной работы
для Deep Learning School (DLS).

В данной работе автором были выполнены следующие действия:
1. Подготовлен датасет, содержащий изображения со спутника Земли и маски
2. Реализована кастомная архитектура U-Net для задачи сегментации
"""

LIMITATIONS = """
**Ограничения.** Во время обучения были использованы изображения, содержащие
недостаточное количество информации для обучения модели. Поэтому метрика Dice,
показывающая точность предсказания масок, составляет около 0.08. В дальнейшем
планируется улучшение нейронной сети и ее архитектуры.
"""


@st.cache_resource
def get_model():
    """Load the model once and reuse it between reruns."""
    return load_model(MODEL_PATH)


def show_legend() -> None:
    st.subheader("Легенда классов маски сегментации")
    cols = st.columns(4)
    for i, cls in enumerate(CLASSES):
        with cols[i % 4]:
            st.color_picker(
                label=f"Класс {cls.id}: {cls.name}",
                value=cls.color,
                disabled=True,
                key=f"legend_{cls.id}",
            )


def show_error(error: Exception) -> None:
    st.error(f"Ошибка: {error}")
    st.info("Убедитесь, что:")
    st.write(f"- Файл модели лежит в папке с приложением под именем '{MODEL_PATH.name}'")
    st.write("- Файл модели скачан через Git LFS (а не только указатель на него)")
    st.write("- Архитектура модели совпадает с обученной")


def main() -> None:
    st.title(TITLE)
    st.write(INTRO)

    with st.sidebar:
        show_legend()

    uploaded_file = st.file_uploader("Выберите спутниковый снимок", type=SUPPORTED_IMAGE_TYPES)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col_input, col_mask = st.columns(2)

        with col_input:
            st.image(image, caption="Исходное изображение", use_container_width=True)

        try:
            mask = predict_mask(get_model(), image)
        except Exception as error:  # show a friendly message in the UI
            show_error(error)
        else:
            with col_mask:
                st.image(colorize_mask(mask), caption="Предсказанная маска", use_container_width=True)

    st.info(LIMITATIONS)


if __name__ == "__main__":
    main()
