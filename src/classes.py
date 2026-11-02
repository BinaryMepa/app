"""Segmentation classes: a single source of truth for names and colors.

Both the predicted mask and the legend use these colors, so they always match.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentationClass:
    id: int
    name: str
    color: str  # HEX, e.g. "#FFB6C1"


CLASSES: tuple[SegmentationClass, ...] = (
    SegmentationClass(0, "Фон / Не классифицировано", "#DCDCDC"),
    SegmentationClass(1, "Жилые здания", "#FFB6C1"),
    SegmentationClass(2, "Коммерческие здания", "#90EE90"),
    SegmentationClass(3, "Дороги", "#ADD8E6"),
    SegmentationClass(4, "Водоемы", "#98FB98"),
    SegmentationClass(5, "Леса", "#DDA0DD"),
    SegmentationClass(6, "Сельхозугодья", "#E0FFFF"),
    SegmentationClass(7, "Пустующие земли", "#FA8072"),
    SegmentationClass(8, "Транспортная инфраструктура", "#D3D3D3"),
    SegmentationClass(9, "Парки и зоны отдыха", "#FFDAB9"),
    SegmentationClass(10, "Промышленные зоны", "#FFFF00"),
    SegmentationClass(11, "Мосты", "#00FFFF"),
    SegmentationClass(12, "Железные дороги", "#FF00FF"),
    SegmentationClass(13, "Аэропорты", "#808000"),
    SegmentationClass(14, "Спортивные сооружения", "#008080"),
    SegmentationClass(15, "Пляжи", "#800080"),
    SegmentationClass(16, "Горы и холмы", "#00FF7F"),
    SegmentationClass(17, "Реки и каналы", "#FF6347"),
    SegmentationClass(18, "Городские парковки", "#0000FF"),
)

NUM_CLASSES = len(CLASSES)
