"""
Roda o modelo Mask R-CNN treinado em UMA imagem e salva o resultado com as
bounding boxes e os nomes das classes desenhados.

Uso (dentro do container):

    # usa automaticamente o melhor modelo mais recente em logs/
    python samples/custom/eval_single_image.py --image /workspace/Dataset/valid/foo.jpg

    # apontando um checkpoint especifico e a saida
    python samples/custom/eval_single_image.py \
        --image /workspace/Dataset/valid/foo.jpg \
        --weights logs/custom20260630T0226/best_custom.h5 \
        --output /workspace/Mask_RCNN/foo_pred.jpg

--weights aceita um .h5 ou 'last' (default: best_custom.h5 mais recente em logs/).
--dataset (ou a env DATASET_PATH) e usado apenas para ler os NOMES das classes
(precisa ser o mesmo dataset do treino para a ordem das classes bater).
"""

import os
import sys
import argparse

import numpy as np
import skimage.io
import skimage.color
import cv2

# Raiz do repo, relativa a este arquivo (funciona de qualquer cwd)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(__file__))  # para importar custom.py

from mrcnn import model as modellib
from custom import build_config, CustomDataset

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def find_latest_best(logs_dir):
    """Retorna o best_custom.h5 mais recente em logs/ (fallback: qualquer .h5)."""
    best, others = [], []
    for root, _, files in os.walk(logs_dir):
        for f in files:
            if not f.endswith(".h5"):
                continue
            p = os.path.join(root, f)
            (best if f == "best_custom.h5" else others).append((os.path.getmtime(p), p))
    pool = best or others
    return max(pool)[1] if pool else None


def color_for(class_id):
    """Cor deterministica (BGR) por classe."""
    v = (int(class_id) * 1234567) % 0xFFFFFF
    return (v & 255, (v >> 8) & 255, (v >> 16) & 255)


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia Mask R-CNN em uma imagem (bbox + nome das classes).")
    parser.add_argument("--image", required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--weights", default="last",
                        help="Arquivo .h5 ou 'last' (default: best mais recente em logs/)")
    parser.add_argument("--dataset", default=os.environ.get("DATASET_PATH"),
                        help="Raiz do dataset (so para os nomes das classes; default DATASET_PATH)")
    parser.add_argument("--output", default=None,
                        help="Caminho de saida (default: <imagem>_pred.jpg)")
    parser.add_argument("--logs", default=DEFAULT_LOGS_DIR,
                        help="Diretorio de logs (para resolver --weights last)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Score minimo para mostrar uma deteccao (override da config)")
    args = parser.parse_args()

    assert args.dataset, "Defina --dataset ou DATASET_PATH (necessario para os nomes das classes)"
    assert os.path.isfile(args.image), "Imagem nao encontrada: {}".format(args.image)

    # Nomes das classes a partir do dataset (index 0 = BG)
    dataset = CustomDataset()
    dataset.load_custom(args.dataset, "train")
    dataset.prepare()
    class_names = dataset.class_names
    num_fg = len(class_names) - 1

    # Config de inferencia (mesmas dimensoes do treino)
    config = build_config(num_fg, inference=True)
    if args.confidence is not None:
        config.DETECTION_MIN_CONFIDENCE = args.confidence

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    weights = args.weights
    if weights == "last":
        weights = find_latest_best(args.logs)
        assert weights, "Nenhum .h5 encontrado em {}".format(args.logs)
    print("Pesos:", weights)
    model.load_weights(weights, by_name=True)

    # Le a imagem em RGB
    image = skimage.io.imread(args.image)
    if image.ndim == 2:
        image = skimage.color.gray2rgb(image)
    if image.shape[-1] == 4:
        image = image[..., :3]

    r = model.detect([image], verbose=0)[0]
    n = r["rois"].shape[0]
    print("Deteccoes:", n)

    # Desenha as caixas (em BGR para o OpenCV)
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(n):
        y1, x1, y2, x2 = (int(v) for v in r["rois"][i])
        cid = int(r["class_ids"][i])
        score = float(r["scores"][i])
        label = "{} {:.2f}".format(class_names[cid], score)
        color = color_for(cid)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y1 - th - bl - 4, 0)
        cv2.rectangle(vis, (x1, top), (x1 + tw, top + th + bl + 4), color, -1)
        cv2.putText(vis, label, (x1, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out = args.output or (os.path.splitext(args.image)[0] + "_pred.jpg")
    cv2.imwrite(out, vis)
    print("Salvo em:", out)


if __name__ == "__main__":
    main()
