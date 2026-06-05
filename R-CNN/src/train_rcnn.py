import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CocoDetectionDataset(Dataset):
    def __init__(self, img_folder, ann_file):
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.img_folder = img_folder

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        raw_file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        clean_file_name = os.path.basename(raw_file_name)
        
        full_path = os.path.join(self.img_folder, clean_file_name)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Arquivo não encontrado em: {full_path}. "
                                    f"Verifique se o arquivo existe dentro de {self.img_folder}")

        img = Image.open(full_path).convert("RGB")
        
        boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in [obj['bbox'] for obj in anns]]
        labels = [obj['category_id'] for obj in anns]
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }
        
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) 
        
        return img_tensor, target

    def __len__(self):
        return len(self.ids)

def find_json_file(directory, pattern):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and pattern in file.lower():
                return os.path.join(root, file)
    raise FileNotFoundError(f"JSON não encontrado em {directory}")

# Execução
dataset_base = "/workspace/Dataset/MobIA 5.4.coco" 

# Busca o arquivo JSON dentro da pasta de anotações (ou onde ele estiver)
train_json = find_json_file(dataset_base, "annotations") 

# A pasta de imagens é a pasta 'train' que fica no mesmo nível das 'annotations'
train_img_folder = os.path.join(dataset_base, "train")

print(f"DEBUG: Buscando imagens em: {train_img_folder}")
print(f"DEBUG: Buscando JSON em: {train_json}")

train_ds = CocoDetectionDataset(train_img_folder, train_json)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
device = torch.device('cuda')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 7)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()
for epoch in range(10):
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Loss: {losses.item()}")