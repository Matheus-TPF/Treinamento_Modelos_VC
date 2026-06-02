import os
import torch
import cv2

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer


class CocoDetectionDataset(Dataset):
    """Dataset personalizado para carregar dados no formato COCO para o DETR.

    Esta classe faz a ponte entre as anotações brutas do padrão COCO e o formato
    de dicionário esperado pelos modelos de detecção da Hugging Face.

    Attributes:
        coco (COCO): Instância da API COCO para gerenciar as anotações.
        ids (list of int): Lista com todos os IDs únicos de imagens do dataset.
        img_folder (str): Caminho para o diretório que contém as imagens.
        processor (DetrImageProcessor): Processador do DETR para transformação
            e normalização das imagens.
    """

    def __init__(self, img_folder: str, ann_file: str, processor: DetrImageProcessor):
        """Inicializa o dataset carregando as anotações COCO.

        Args:
            img_folder (str): Caminho para a pasta de imagens (ex: 'train' ou 'valid').
            ann_file (str): Caminho para o arquivo JSON de anotações COCO.
            processor (DetrImageProcessor): Processador oficial do modelo DETR.
        """
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.img_folder = img_folder
        self.processor = processor

    def __len__(self) -> int:
        """Retorna o número total de imagens do Dataset.

        Returns: 
            int: Quantidade de amostras disponíveis.
        """
        return len(self.ids)
    
    def __getitem__(self, index: int) -> dict:
        """Carrega uma imagem e suas respectivas anotações convertidas para o DETR.

        O método lê a imagem do disco, converte seu espaço de cores, extrai
        as caixas delimitadoras (bounding boxes) do formato COCO e delega ao
        `processor` a tarefa de aplicar as normalizações necessárias.

        Args:
            index (int): Índice da amostra a ser carregada.

        Returns:
            dict: Dicionário contendo os tensores processados estruturado como:
                - "pixel_values" (torch.Tensor): Tensor da imagem normalizada.
                - "labels" (dict): Dicionário com chaves 'boxes', 'class_labels', etc.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.img_folder, path)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        boxes = []
        area = []
        iscrowd = []
        category_id = []

        for ann in coco_annotation:
            boxes.append(ann['bbox'])
            area.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
            category_id.append(ann['category_id'])
            
        target = {
            "image_id": torch.tensor([img_id]),
            "annotations": {
                "bbox": boxes,
                "category_id": category_id,
                "area": area,
                "iscrowd": iscrowd
            }
        }
        
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"][0]
        labels = encoding["labels"][0]
        
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch: list) -> dict:
    """Agrupa e alinha um lote de amostras antes de enviá-lo para a GPU.

    Diferente de tarefas simples como classificação, o DETR exige que as
    imagens sejam empilhadas e as anotações (que variam de tamanho por imagem)
    sejam mantidas como uma lista de dicionários dentro do lote final.

    Args:
        batch (list of dict): Lista de dicionários retornados individualmente
            pelo método `__getitem__` do Dataset.

    Returns:
        dict: Lote final pronto para o modelo contendo:
            - "pixel_values" (torch.Tensor): Tensor 4D de imagens agrupadas.
            - "labels" (list of dict): Lista contendo as anotações de cada imagem.
    """
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_values = torch.stack(pixel_values)
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


# Execução propriamente

# 1. Inicializa o processador de imagem oficial
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 2. Instancia os Datasets usando a sua classe customizada e caminhos corretos
dataset_base_dir = "/workspace/Dataset/MobIA 5.4.coco"

train_dataset = CocoDetectionDataset(
    img_folder=os.path.join(dataset_base_dir, "train"),
    ann_file=os.path.join(dataset_base_dir, "annotations", "train_annotations.coco.json"),
    processor=processor
)

val_dataset = CocoDetectionDataset(
    img_folder=os.path.join(dataset_base_dir, "valid"),
    ann_file=os.path.join(dataset_base_dir, "annotations", "valid_annotations.coco.json"),
    processor=processor
)

# 3. Carrega o modelo 
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=6,
    ignore_mismatched_sizes=True
)

# 4. Configuração de hiperparâmetros
training_args = TrainingArguments(
    output_dir="/workspace/outputs",
    per_device_train_batch_size=4,
    num_train_epochs=15,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    remove_unused_columns=False
)

# 5. Configura o Trainer da Hugging Face injetando seus componentes estruturados
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

# 6. Começa o treinamento
trainer.train()