"""
Mask R-CNN
Treino em um dataset custom no formato COCO (mesmo padrao usado pelo
DETR/YOLOX deste projeto: pastas de imagens + um .json COCO por split).

Layout esperado do dataset (apontado por DATASET_PATH / --dataset):

    $DATASET_PATH/
        train/              <- imagens de treino
        train_coco.json     <- anotacoes COCO do treino
        valid/              <- imagens de validacao
        valid_coco.json     <- anotacoes COCO da validacao

Uso (linha de comando):

    # Treinar a partir dos pesos pre-treinados de COCO
    python3 samples/custom/custom.py train --dataset=/workspace/Dataset --weights=coco

    # Treinar a partir do ImageNet
    python3 samples/custom/custom.py train --dataset=/workspace/Dataset --weights=imagenet

    # Continuar o ultimo treino
    python3 samples/custom/custom.py train --weights=last

    # Ajustando o early stopping (mAP@0.5:0.95 de validacao)
    python3 samples/custom/custom.py train --weights=coco \
        --epochs 100 --patience 20 --min-delta 0.001

Se --dataset nao for passado, usa a variavel de ambiente DATASET_PATH
(definida no .env, veja .env_example).

Early stopping: ao fim de cada epoca o script mede o mAP@0.5:0.95 no conjunto
de validacao (pycocotools) e para o treino se nao houver melhora por
--patience epocas. O melhor modelo e salvo em logs/<run>/best_custom.h5.
"""

import os
import sys
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*`np\.bool`.*")

import keras
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

ANN_TEMPLATE = "{subset}_coco.json"


############################################################
#  Configuracao
############################################################

def build_config(num_classes, images_per_gpu=2, steps_per_epoch=None, inference=False):
    """Cria a config com o numero certo de classes.

    NUM_CLASSES precisa ser definido ANTES de instanciar a Config (a base
    calcula IMAGE_META_SIZE a partir dele no __init__), por isso usamos uma
    fabrica em vez de uma classe estatica.

    num_classes: numero de classes de primeiro plano (sem o fundo).
    """

    class CustomConfig(Config):
        NAME = "custom"
        IMAGES_PER_GPU = 1 if inference else images_per_gpu
        GPU_COUNT = 1
        NUM_CLASSES = 1 + num_classes
        STEPS_PER_EPOCH = steps_per_epoch or 100
        DETECTION_MIN_CONFIDENCE = 0.7
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512          # multiplo de 64
        TRAIN_ROIS_PER_IMAGE = 100   # padrao 200; metade dos ROIs por imagem
        LOSS_WEIGHTS = {
            "rpn_class_loss": 1.0,
            "rpn_bbox_loss": 1.0,
            "mrcnn_class_loss": 1.0,
            "mrcnn_bbox_loss": 1.0,
            "mrcnn_mask_loss": 0.0,
        }

    return CustomConfig()


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):
    """Carrega um split (train/valid) no formato COCO."""

    def load_custom(self, dataset_dir, subset, return_coco=False):
        ann_path = os.path.join(dataset_dir, ANN_TEMPLATE.format(subset=subset))
        image_dir = os.path.join(dataset_dir, subset)
        coco = COCO(ann_path)

        class_ids = sorted(coco.getCatIds())
        for i in class_ids:
            self.add_class("custom", i, coco.loadCats(i)[0]["name"])

        for i in list(coco.imgs.keys()):
            self.add_image(
                "custom", image_id=i,
                path=os.path.join(image_dir, os.path.basename(coco.imgs[i]["file_name"])),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=None)))

        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Gera as mascaras de instancia de uma imagem.
        Retorna:
            masks: array bool [altura, largura, n_instancias]
            class_ids: array 1D com os IDs de classe de cada instancia
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(CustomDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        for ann in image_info["annotations"]:
            class_id = self.map_source_class_id("custom.{}".format(ann["category_id"]))
            if not class_id:
                continue
            m = self.annToMask(ann, image_info["height"], image_info["width"])
            if m.max() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            return mask, np.array(class_ids, dtype=np.int32)
        return super(CustomDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        return super(CustomDataset, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        segm = ann["segmentation"]
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm["counts"], list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        return maskUtils.decode(rle)


def count_foreground_classes(dataset_dir, subset):
    """Le o numero de categorias do split para dimensionar NUM_CLASSES."""
    ann_path = os.path.join(dataset_dir, ANN_TEMPLATE.format(subset=subset))
    return len(COCO(ann_path).getCatIds())


def count_images(dataset_dir, subset):
    """Numero de imagens do split (para cobrir o dataset inteiro por epoca)."""
    ann_path = os.path.join(dataset_dir, ANN_TEMPLATE.format(subset=subset))
    return len(COCO(ann_path).getImgIds())


def evaluate_coco_map(inference_model, dataset, coco_gt, limit=0):
    """Roda inferencia no dataset e retorna o mAP@[.5:.95] (bbox), a mesma
    metrica usada no early stopping do DETR e do YOLOX."""
    image_ids = dataset.image_ids
    if limit:
        image_ids = image_ids[:limit]

    results = []
    evaluated_ids = []
    total = len(image_ids)
    for n, image_id in enumerate(image_ids, 1):
        image = dataset.load_image(image_id)
        r = inference_model.detect([image], verbose=0)[0]
        print("\r  [val] {}/{} imagens avaliadas".format(n, total), end="", flush=True)
        src_id = dataset.image_info[image_id]["id"]  # id original do COCO
        evaluated_ids.append(src_id)
        for i in range(r["rois"].shape[0]):
            y1, x1, y2, x2 = r["rois"][i]
            results.append({
                "image_id": src_id,
                "category_id": dataset.get_source_class_id(r["class_ids"][i], "custom"),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(r["scores"][i]),
            })

    print() 

    if not results:  
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = evaluated_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])  # stats[0] = AP@[.5:.95]


class CocoMapEarlyStopping(keras.callbacks.Callback):
    """Ao fim de cada epoca, mede o mAP@0.5:0.95 no val, salva o melhor
    checkpoint e para o treino apos `patience` epocas sem melhora.

    Mesma logica de early stopping do DETR/YOLOX deste projeto.
    """

    def __init__(self, mrcnn_model, inference_model, dataset_val, coco_gt,
                 best_path, patience=20, min_delta=0.001, eval_limit=0):
        super(CocoMapEarlyStopping, self).__init__()
        self.mrcnn_model = mrcnn_model          # wrapper MaskRCNN (para checkpoint_path)
        self.inference_model = inference_model  # modelo em modo inference
        self.dataset_val = dataset_val
        self.coco_gt = coco_gt
        self.best_path = best_path
        self.patience = patience
        self.min_delta = min_delta
        self.eval_limit = eval_limit
        self.best_map = -1.0
        self.best_epoch = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        ckpt = self.mrcnn_model.checkpoint_path.format(epoch=epoch + 1)
        self.inference_model.load_weights(ckpt, by_name=True)
        mAP = evaluate_coco_map(self.inference_model, self.dataset_val,
                                self.coco_gt, self.eval_limit)
        print("\n[val] epoch {} - mAP@0.5:0.95 = {:.4f} "
              "(melhor: {:.4f} @ epoch {})".format(
                  epoch + 1, mAP, max(self.best_map, 0.0), self.best_epoch))

        if mAP - self.best_map > self.min_delta:
            self.best_map = mAP
            self.best_epoch = epoch + 1
            self.wait = 0
            self.inference_model.keras_model.save_weights(self.best_path)
            print("[val] novo melhor mAP -> salvo em {}".format(self.best_path))
        else:
            self.wait += 1
            print("[val] sem melhora ({}/{})".format(self.wait, self.patience))
            if self.patience and self.wait >= self.patience:
                print("[early stopping] {} epocas sem melhora. Parando.".format(self.wait))
                self.model.stop_training = True


class ImageCounter(keras.callbacks.Callback):
    """Mostra quantas imagens ja foram processadas na epoca atual.

    Total por epoca = STEPS_PER_EPOCH x BATCH_SIZE (pode ser menor que o
    dataset inteiro, dependendo de STEPS_PER_EPOCH).
    """

    def __init__(self, batch_size, steps_per_epoch):
        super(ImageCounter, self).__init__()
        self.batch_size = batch_size
        self.total = steps_per_epoch * batch_size
        self.count = 0
        self.cur_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.count = 0
        self.cur_epoch = epoch + 1

    def on_batch_end(self, batch, logs=None):
        self.count += (logs or {}).get("size", self.batch_size)
        print("\r  epoch {}: {}/{} imagens processadas".format(
            self.cur_epoch, self.count, self.total), end="", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        print()  # quebra a linha do contador antes da avaliacao de mAP


def train(model, dataset_dir, epochs=100, patience=20, min_delta=0.001,
          layers="heads", eval_limit=0):
    dataset_train = CustomDataset()
    dataset_train.load_custom(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    coco_gt = dataset_val.load_custom(dataset_dir, "valid", return_coco=True)
    dataset_val.prepare()

    num_fg = len(dataset_train.class_info) - 1  # menos o background
    inference_model = modellib.MaskRCNN(
        mode="inference",
        config=build_config(num_fg, inference=True),
        model_dir=model.model_dir)

    best_path = os.path.join(model.log_dir, "best_custom.h5")
    early_stop = CocoMapEarlyStopping(
        mrcnn_model=model, inference_model=inference_model,
        dataset_val=dataset_val, coco_gt=coco_gt, best_path=best_path,
        patience=patience, min_delta=min_delta, eval_limit=eval_limit)

    img_counter = ImageCounter(model.config.BATCH_SIZE, model.config.STEPS_PER_EPOCH)

    print("Treinando '{}' ate {} epocas (early stopping: patience={}, "
          "min_delta={})".format(layers, epochs, patience, min_delta))
    model.train(dataset_train, dataset_val,
                learning_rate=model.config.LEARNING_RATE,
                epochs=epochs,
                layers=layers,
                custom_callbacks=[img_counter, early_stop])

    print("\nMelhor mAP@0.5:0.95 = {:.4f} (epoch {})".format(
        early_stop.best_map, early_stop.best_epoch))
    print("Melhor modelo salvo em: {}".format(best_path))


############################################################
#  Main
############################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treina Mask R-CNN em dataset custom (COCO).")
    parser.add_argument("command", metavar="<command>", help="'train'")
    parser.add_argument("--dataset", required=False,
                        default=os.environ.get("DATASET_PATH"),
                        metavar="/path/to/dataset/",
                        help="Raiz do dataset (default: variavel de ambiente DATASET_PATH)")
    parser.add_argument("--weights", required=True,
                        metavar="/path/to/weights.h5",
                        help="Caminho do .h5, ou 'coco' / 'imagenet' / 'last'")
    parser.add_argument("--logs", required=False, default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help="Diretorio de logs/checkpoints (default: logs/)")
    parser.add_argument("--batch", required=False, default=2, type=int,
                        help="Imagens por GPU = batch de treino (BATCH_SIZE). Reduza para 1 se faltar VRAM")
    parser.add_argument("--steps", required=False, default=0, type=int,
                        help="Passos por epoca (0 = auto: cobre todas as imagens de treino)")
    parser.add_argument("--epochs", required=False, default=100, type=int,
                        help="Teto de epocas (early stopping costuma parar antes)")
    parser.add_argument("--patience", required=False, default=20, type=int,
                        help="Epocas sem melhora no mAP de val antes de parar (0 desativa)")
    parser.add_argument("--min-delta", required=False, default=0.001, type=float,
                        dest="min_delta",
                        help="Ganho minimo de mAP para contar como melhora")
    parser.add_argument("--layers", required=False, default="heads",
                        help="Camadas a treinar: heads | all | 3+ | 4+ | 5+ (default: heads)")
    parser.add_argument("--eval-limit", required=False, default=0, type=int,
                        dest="eval_limit",
                        help="Limita o n de imagens de val na avaliacao (0 = todas)")
    args = parser.parse_args()

    assert args.command == "train", "Comando suportado: 'train'"
    assert args.dataset, "Defina --dataset ou a variavel de ambiente DATASET_PATH (.env)"

    print("Weights:", args.weights)
    print("Dataset:", args.dataset)
    print("Logs:", args.logs)

    num_fg = count_foreground_classes(args.dataset, "train")
    print("Classes de primeiro plano detectadas:", num_fg)
    config = build_config(num_fg, images_per_gpu=args.batch)
    if args.steps:
        config.STEPS_PER_EPOCH = args.steps
    else:
        n_train = count_images(args.dataset, "train")
        config.STEPS_PER_EPOCH = math.ceil(n_train / config.BATCH_SIZE)
    print("Imagens de treino: {} | batch: {} | steps/epoca: {}".format(
        count_images(args.dataset, "train"), config.BATCH_SIZE, config.STEPS_PER_EPOCH))
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Carregando pesos:", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    train(model, args.dataset,
          epochs=args.epochs, patience=args.patience, min_delta=args.min_delta,
          layers=args.layers, eval_limit=args.eval_limit)
