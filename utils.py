import os
import random
from datetime import datetime
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from dataset import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            # out_indices=[5,6]
        )

        # self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.HIDDEN_DIM)
        # self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        # self.mass_layer = nn.Linear(1, config.HIDDEN_DIM)
        self.regressor = nn.Sequential(
            nn.Linear(self.image_model.num_features + 1 + config.NUM_CLASSES, config.NUM_CLASSES),
            # nn.LayerNorm(config.NUM_CLASSES),
            # nn.BatchNorm1d(config.NUM_CLASSES),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            # nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES),
            nn.Linear(config.NUM_CLASSES, 1)      
        )

    def forward(self, input_ids, attention_mask, image, mass, text_id):
        # text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        # text_emb = self.text_proj(text_features)
        # mass_emb = self.mass_layer(mass)
        # image_emb = self.image_proj(image_features)

        # fused_emb = image_emb * mass_emb # * text_emb
        fused_emb = torch.cat([image_features, mass, text_id], dim=1)
        mass = self.regressor(fused_emb)
        return mass


def train(config, device, n_rows=None):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # set_requires_grad(model.text_model,
    #                   unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([
    # {
    #     'params': model.text_model.parameters(),
    #     'lr': config.TEXT_LR
    # }, 
    {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, 
    {
        'params': model.regressor.parameters(),
        'lr': config.REGRESSOR_LR
    }])

    criterion = nn.L1Loss(reduction="mean")

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultimodalDataset(config, transforms, n_rows=n_rows)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type="test", n_rows=n_rows)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    # инициализируем метрику
    mae_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    best_mae_val = float('inf')

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].unsqueeze(1).to(device),
                "text_id": batch['text_id'].to(device)
            }
            targets = batch['target'].unsqueeze(1).to(device)

            # Forward
            optimizer.zero_grad()
            predicts = model(**inputs)
            loss = criterion(predicts, targets)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _ = mae_metric_train(preds=predicts, target=targets)

        # Валидация
        train_mae = mae_metric_train.compute().cpu().numpy()
        val_mae = validate(model, val_loader, device, mae_metric_val)
        mae_metric_val.reset()
        mae_metric_train.reset()

        print(
            f"{datetime.now()} Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae :.4f}| Val MAE: {val_mae :.4f}"
        )

        if val_mae < best_mae_val:
            print(f"Save best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
        if val_mae < config.TARGET_MAE:
            break
    print("Train ended")

def validate(model, val_loader, device, metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].unsqueeze(1).to(device),
                "text_id": batch['text_id'].to(device)
            }
            targets = batch['target'].unsqueeze(1).to(device)

            predicts = model(**inputs)
            _ = metric(preds=predicts, target=targets)

    return metric.compute().cpu().numpy()
