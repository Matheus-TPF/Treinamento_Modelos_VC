import os
import torch
import cv2

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

dataset_base_dir = "/workspace/Dataset/MobIA 5.4.coco"
dataset = load_dataset("imagefolder", data_files= {
    "train": os.path.join(dataset_base_dir, "annotations" ,"train_annotations.coco.json"),
    "validation": os.path.join(dataset_base_dir, "annotations" ,"valid_annotations.coco.json")
})

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=6,
    ignore_mismatched_sizes=True
)

# Configuração de hiperparametros 
training_args = TrainingArguments(
    output_dir = "/workspace/outputs",
    per_device_train_batch_size=4,
    num_train_epochs=15,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    remove_unused_columns=False
)

def collate_fn(batch):
    pixel_values= [processor(images=x["image"], return_tensors="pt")["pixel_values"][0] for x in batch]
    pixel_values = torch.stack(pixel_values)
    labels = [x["conditioning"] for x in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# Configura o treinamento

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= dataset["train"],
    eval_dataset= dataset["validation"],
    data_collator=collate_fn
)

# Começa o treinamento
trainer.train()