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

2. Entre na pasta do YoloX

```bash
cd .\YOLOX\  
```

3. Build o container

```bash
docker compose build yolox
```


4. Executar o treinamento
```bash
docker compose up yolox
```

#### DETR

1. Entre na pasta específica do DETR

```bash
cd .\DETR\  
```

2. Buildar o container específico do DETR
```bash
docker compose build detr
```

3. Rodar o treinamento
```bash
docker compose up detr
```

#### R-CNN

