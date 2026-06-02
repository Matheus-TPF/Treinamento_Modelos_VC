# Para executar:

#### YOLOX
1. Baixe o dataset

1.1 Crie uma pasta annotations e mova os ``_annotations.coco.json`` de treinamento e validação para lá

Dessa forma: 
```
├───Dataset
   └───MobIA 5.4.coco
       ├───annotations
       ├───train
       └───valid
```
1.2 Renomeie o seu ``_annotations.coco.json`` de treinamento e validação para: 

```
train_annotations.coco.json

valid_annotations.coco.json
```

**OBS:** Essa configuração é valida tanto para YoloX quanto para DETR

2. Entre na pasta e execute o docker

```bash
cd .\YOLOX\  
```

```bash
docker compose run yolox
```

# Comandos Docker:

Para buildar o container:

```bash
docker compose build --no-cache
```

Para rodar o treinamento:
```bash
docker compose up
```

``-f caminho_configuracao_yolox`` para usar a configuração específica do treinamento costumizado,
``-d 1`` para usar GPU,  
``-b 16`` para o número definir 16 imagens por lote

#### DETR

Buildar o container específico do DETR
```bash
docker compose build detr
```

Rodar o treinamento
```bash
docker compose up detr
```

#### R-CNN

