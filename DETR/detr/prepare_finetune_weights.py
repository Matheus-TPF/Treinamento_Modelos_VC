"""Prepara os pesos do DETR-R50 (pre-treinado no COCO) para fine-tuning.

Remove a cabeca de classificacao (class_embed.*) e o estado de treino
(optimizer/lr_scheduler/epoch), deixando so os pesos do backbone/transformer.
Assim o modelo carrega tudo com strict=False e reinicializa apenas a class_embed
para o novo numero de classes do dataset.

Uso (dentro do container):
    python prepare_finetune_weights.py --output weights/detr-r50_finetune.pth
"""
import argparse
import torch

# Checkpoints oficiais (91 classes do COCO).
URLS = {
    "r50": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
    "r101": "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=URLS.keys(), default="r50",
                        help="Backbone do checkpoint base (r50 ou r101)")
    parser.add_argument("--url", default=None,
                        help="URL ou caminho de um checkpoint base (sobrescreve --model)")
    parser.add_argument("--output", default=None,
                        help="Saida; padrao: weights/detr-<model>_finetune.pth")
    args = parser.parse_args()

    source = args.url or URLS[args.model]
    output = args.output or f"weights/detr-{args.model}_finetune.pth"

    if source.startswith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(source, map_location="cpu")

    state_dict = checkpoint["model"]
    removed = [k for k in list(state_dict.keys()) if k.startswith("class_embed")]
    for k in removed:
        del state_dict[k]

    print(f"Removidas {len(removed)} chaves de class_embed: {removed}")
    torch.save({"model": state_dict}, output)
    print(f"Checkpoint para fine-tuning salvo em: {output}")


if __name__ == "__main__":
    main()
