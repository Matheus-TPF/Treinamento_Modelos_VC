import os
import torch
import cv2
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
        
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        
        boxes = [obj['bbox'] for obj in anns]
        # Converte de [x, y, w, h] para [x1, y1, x2, y2]
        boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes]
        
        labels = [obj['category_id'] for obj in anns]
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }
        
        # Converte imagem para tensor (0 a 1)
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32) / 255.0
        img_tensor = img_tensor.permute(2, 0, 1) # HWC to CHW
        
        return img_tensor, target

    def __len__(self):
        return len(self.ids)

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def find_json_file(directory: str, pattern: str) -> str:
    """Busca dinamicamente um arquivo JSON que combine com o padrão fornecido.

    Args:
        directory (str): Diretório base para a varredura.
        pattern (str): Palavra-chave contida no nome do arquivo (ex: 'train').

    Returns:
        str: Caminho absoluto do arquivo encontrado.

    Raises:
        FileNotFoundError: Se nenhum arquivo corresponder aos critérios.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and pattern in file.lower():
                return os.path.join(root, file)
    raise FileNotFoundError(f"Nenhum arquivo JSON contendo '{pattern}' foi encontrado em {directory}")

# Execução

dataset_base = "/workspace/Dataset"
train_ds = CocoDetectionDataset(os.path.join(dataset_base, "train"), find_json_file(dataset_base, "train"))
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=7) # 6 classes + background
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()
for epoch in range(10):
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Loss: {losses.item()}")