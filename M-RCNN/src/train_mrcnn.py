import os
import torch
import numpy as np
import torchvision

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
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
        
        img_info = self.coco.loadImgs(img_id)[0]
        full_path = os.path.join(self.img_folder, os.path.basename(img_info["file_name"]))
        img = Image.open(full_path).convert("RGB")
        
        boxes, labels, masks = [], [], []
        for obj in anns:
            if 'segmentation' not in obj or not obj['segmentation'] or obj['bbox'][2] < 1 or obj['bbox'][3] < 1:
                continue
            boxes.append([obj["bbox"][0], obj["bbox"][1], obj["bbox"][0]+obj['bbox'][2], obj["bbox"][1]+obj['bbox'][3]])
            labels.append(obj['category_id'])
            masks.append(self.coco.annToMask(obj))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).view(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8) if len(masks) > 0 else torch.zeros((0, img.height, img.width), dtype=torch.uint8),
            "image_id": torch.tensor([img_id])
        }
        
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img_tensor, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# Execução
dataset_base = "/workspace/Dataset"
train_ds = CocoDetectionDataset(os.path.join(dataset_base, "train"), os.path.join(dataset_base, "annotations/train_annotations.coco.json"))
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
device = torch.device('cuda')

# Modelo
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 8)
model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 8)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

model.train()
for epoch in range(10):
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(f"Loss: {losses.item()}")