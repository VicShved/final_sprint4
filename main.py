import torch
from dataset import MultimodalDataset, get_transforms
from utils import train


class Config:
    # для воспроизводимости
    SEED = 42

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"

    # Гиперпараметры
    BATCH_SIZE = 128
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    EPOCHS = 30
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 4

    # Пути
    DF_PATH = "data/dish.csv"
    SAVE_PATH = "best_model.pth"


device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

transforms = get_transforms(config=config)
train_dataset = MultimodalDataset(config, transforms)
print(train_dataset.df.head())
train_dataset[1]

# for i in range(len(train_dataset)):
#     train_dataset[i]

from dataset import plot_image

row = train_dataset[len(train_dataset) - 1]
plot_image(row['source_image'], row['image'])

# train(config, device)