import streamlit as st
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            DoubleConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        nn.BatchNorm2d(out_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        # Decoder
        self.up0 = Up(2048, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up0(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# Streamlit app
st.title('Модель для сегментации спутниковых снимков')

st.write("""
Здравствуйте! Данное приложение было создано с целью демонстрации дипломной работы для Deep Learning School (DLS).

В данной работе автором были выполнены следующие действия:
1. Подготовлен датасет, содержащий изображения со спутника земли и маски
2. Реализована кастомная архитектура U-Net для задачи сегментации
""")

@st.cache_resource
def load_model(model_path):
    model = UNet(n_classes=19) 
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def process_image(image):
    transform = T.Compose([  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((512, 512)) 
    ])
    return transform(image).unsqueeze(0)

def colorsize_mask(mask, n_classes=19):
    colors = np.array([
    [220, 220, 220], 
    [255, 182, 193], 
    [144, 238, 144], 
    [173, 216, 230],
    [152, 251, 152], 
    [221, 160, 221], 
    [224, 255, 255], 
    [250, 128, 114],
    [211, 211, 211], 
    [255, 218, 185], 
    [255, 255, 0], 
    [0, 255, 255],
    [255, 0, 255],
    [128, 128, 0], 
    [0, 128, 128], 
    [128, 0, 128],
    [0, 255, 127], 
    [255, 99, 71], 
    [0, 0, 255],
])
    # Нормализация цветов для отображения
    return (colors[mask] / 255.0).astype(np.float32)  

# Определяем описание классов и цвета в HEX-формате
# Словарь: описание классов и цвета в HEX-формате
class_descriptions = {
    0: ("Фон / Не классифицировано", "#A39191"),
    1: ("Жилые здания", "#FFB6C1"),
    2: ("Коммерческие здания", "#90EE90"),
    3: ("Дороги", "#ADD8E6"),
    4: ("Водоемы", "#98FB98"),
    5: ("Леса", "#DDA0DD"),
    6: ("Сельхозугодья", "#E0FFFF"),
    7: ("Пустующие земли", "#FA8072"),
    8: ("Транспортная инфраструктура", "#D3D3D3"),
    9: ("Парки и зоны отдыха", "#FFDAB9"),
    10: ("Промышленные зоны", "#FFFF00"),
    11: ("Мосты", "#00FFFF"),
    12: ("Железные дороги", "#FF00FF"),
    13: ("Аэропорты", "#808000"),
    14: ("Спортивные сооружения", "#008080"),
    15: ("Пляжи", "#800080"),
    16: ("Горы и холмы", "#00FF7F"),
    17: ("Реки и каналы", "#FF6347"),
    18: ("Городские парковки", "#0000FF")
}

# Функция: RGB кортеж → HEX строка
def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

# Отображение легенды в сайдбаре
def show_legend():
    st.subheader("Легенда классов маски сегментации")
    cols = st.columns(4)

    for i, (class_id, (desc, color)) in enumerate(class_descriptions.items()):
        with cols[i % 4]:
            # Преобразуем к HEX, если передан RGB-кортеж
            if isinstance(color, tuple):
                color = rgb_to_hex(color)

            st.color_picker(
                label=f"Класс {class_id}: {desc}",
                value=color,
                disabled=True
            )
with st.sidebar:
    show_legend()


upload_file = st.file_uploader("Выберите спутниковый снимок", type=["jpg", "png", "tif"])

if upload_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(upload_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Исходное изображение", use_container_width=True)

    try:
        model = load_model("segmentation_model.pth")  
        
        with torch.no_grad():
            input_tensor = process_image(image)
            output = model(input_tensor)
            mask = torch.argmax(output.squeeze(), dim=0).numpy()
            
            colored_mask = colorsize_mask(mask)
            with col2:
                st.image(colored_mask, caption="Предсказанная маска", use_container_width=True, clamp=True)  

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.info("Убедитесь, что:")
        st.write("- Модель лежит в папке с приложением под именем 'segmentation_model.pt'")
        st.write("- Архитектура модели совпадает с обученной")
        st.write("- Изображение можно преобразовать к размеру 512x512")

st.text(
    """ К сожалению, есть одна проблема, которая ставит под сомнение целесообразность данного приложения. Во время обучения были использованы изображения, содержащие недостаточное количество информации для обучения самой модели.
    Таким образом, метрика DIce, указывающая на точность предскзаания моделью масок, составляет около 0.08 для валовых значений. В дальнейшем планируется улучшение данной нейронной сети и ее архитектуры."""
)