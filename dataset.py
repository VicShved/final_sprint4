import torch
from torch.utils.data import Dataset

from PIL import Image
import timm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

import albumentations as A

from matplotlib import pyplot as plt


class MultimodalDataset(Dataset):

    def __init__(self, config, transforms, ds_type="train", n_rows=None):
        self.config = config
        self.df = pd.read_csv(config.DF_PATH)
        # filter train|test
        self.df = self.df[self.df['split']==ds_type]
        self.df.reset_index(inplace=True)
        # for local test
        if n_rows is not None:
            self.df = self.df[:n_rows]
        self.df_ingr = pd.read_csv(config.INGR_DF_PATH, index_col="id")
        self.df['text'] = self.df["ingredients"].map(self._convert_to_ingr)
        self.df['text_ids']  = self.df["ingredients"].map(self._convert_to_ids)
        self.df['target'] = self.df['total_calories'].astype(float)
        self.df['mass'] = self.df['total_mass'].astype(float)
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = transforms


    def _convert_to_ids(self, s: str) -> int:
        s = s.replace("ingr_", "")
        lcl = s.split(";")
        for i in range(len(lcl)):
            lcl[i] = int(lcl[i].lstrip("0"))
        results = [0] * self.config.NUM_CLASSES
        for ind in lcl:
            results[ind -1] = 1
        return results

    def _convert_to_ingr(self, s: str) -> int:
        s = s.replace("ingr_", "")
        lcl = s.split(";")
        for i in range(len(lcl)):
            lcl[i] = self.df_ingr.loc[int(lcl[i].lstrip("0")), "ingr"]
        return ",".join(lcl)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "text"]
        text_ids = self.df.loc[idx, "text_ids"]
        target = self.df.loc[idx, "target"]
        mass = np.float32(self.df.loc[idx, "mass"])
        img_path = self.df.loc[idx, "dish_id"]

        source_image = Image.open(f"data/images/{img_path}/rgb.png")

        image = self.transforms(image=np.array(source_image))["image"]
        return {"target": target, "image": image, "text": text, "source_image": source_image, "mass": mass, "text_ids": text_ids}


def collate_fn(batch, tokenizer):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    targets = torch.FloatTensor([item["target"] for item in batch])
    text_ids = torch.ShortTensor([item["text_ids"] for item in batch])

    tokenized_input = tokenizer(texts,
                                return_tensors="pt",
                                padding="max_length",
                                truncation=True)
    
    masses = torch.FloatTensor([item["mass"] for item in batch])
    return {
        "target": targets,
        "image": images,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "mass": masses,
        "text_id": text_ids,
    }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0,),
                # A.CenterCrop(
                #     height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Affine(
                        # scale=(0.9, 1.1),
                        rotate=(-90, 90),
                        #  translate_percent=(-0.1, 0.1),
                        #  shear=(-10, 10),
                         fill=0,
                         p=0.8),
                # A.CoarseDropout(
                #     num_holes_range=(2, 8),
                #     hole_height_range=(int(0.07 * cfg.input_size[1]),
                #                        int(0.15 * cfg.input_size[1])),
                #     hole_width_range=(int(0.1 * cfg.input_size[2]),
                #                       int(0.15 * cfg.input_size[2])),
                #     fill=0,
                #     p=0.5),
                # A.ColorJitter(brightness=0.2,
                #               contrast=0.2,
                #               saturation=0.2,
                #               hue=0.1,
                #               p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)  #, transpose_mask=True
            ],
            seed=42,
        )
    else:
        transforms = A.Compose(
            [
                A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                # A.CenterCrop(
                #     height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0) # , transpose_mask=True
            ],
            seed=42,
        )

    return transforms


def plot_image(original, transformed, figsize=(5, 2.5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # оригинал
    ax1.imshow(original)
    ax1.set_title('original')
    ax1.axis('off')

    # аугментации
    ax2.imshow(transformed.permute(1, 2, 0).numpy())
    ax2.set_title('transformed')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()