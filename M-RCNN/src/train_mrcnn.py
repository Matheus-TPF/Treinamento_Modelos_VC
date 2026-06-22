import os
import torch
import numpy as np
import torchvision

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
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


@torch.no_grad()
def evaluate_map(model, data_loader, device):
    """Avalia no conjunto de validacao e retorna o mAP@[.5:.95] (bbox),
    a mesma metrica usada no early stopping do DETR e do YOLOX."""
    model.eval()
    coco_gt = data_loader.dataset.coco
    results = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            for box, score, label in zip(output["boxes"].cpu(), output["scores"].cpu(), output["labels"].cpu()):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO usa [x, y, w, h]
                    "score": float(score),
                })
    if not results:  # modelo ainda nao detecta nada -> mAP 0
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])  # stats[0] = AP@[.5:.95]


# Execução
dataset_base = "/workspace/Dataset"
train_ds = CocoDetectionDataset(os.path.join(dataset_base, "train"), os.path.join(dataset_base, "train_coco.json"))
val_ds = CocoDetectionDataset(os.path.join(dataset_base, "valid"), os.path.join(dataset_base, "valid_coco.json"))
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
device = torch.device('cuda')

# Modelo
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 8)
model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 8)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# ---------------- early stopping (mesma logica de DETR/YOLOX) ----------------
EPOCHS = 100           # teto de epocas (early stopping costuma parar antes)
PATIENCE = 2         # epocas sem melhora no mAP de val antes de parar (0 desativa)
MIN_DELTA = 0.001      # ganho minimo de mAP para contar como melhora
OUTPUT_DIR = "/workspace/src/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

best_map = -1.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, n_batches = 0.0, 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        n_batches += 1
    train_loss = running_loss / max(n_batches, 1)

    val_map = evaluate_map(model, val_loader, device)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_mAP={val_map:.4f}")

    if val_map > best_map + MIN_DELTA:
        best_map = val_map
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_mrcnn.pth"))
        print(f"[early stopping] novo melhor mAP={best_map:.4f} -> best_mrcnn.pth")
    else:
        epochs_no_improve += 1
        print(f"[early stopping] sem melhora ha {epochs_no_improve}/{PATIENCE} epocas (melhor mAP={best_map:.4f})")
        if PATIENCE > 0 and epochs_no_improve >= PATIENCE:
            print(f"[early stopping] parando: {PATIENCE} epocas sem melhora. Melhor mAP={best_map:.4f}")
            break
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "last_mrcnn.pth"))
print(f"Treino finalizado. Melhor mAP={best_map:.4f}")
