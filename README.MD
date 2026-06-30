# Para executar:

#### YOLOX

> **Layout do dataset (YOLOX):** o Exp `yolox_s_vehicles.py` espera o dataset em
> `YOLOX/YOLOX/datasets/vehicles/` (no host), montado em
> `/workspace/YOLOX/datasets/vehicles` no container, com as anotações COCO renomeadas
> para `instances_train.json` / `instances_val.json`:
> ```
> YOLOX/YOLOX/datasets/vehicles/
> ├───annotations/
> │   ├───instances_train.json
> │   └───instances_val.json
> ├───train2017/   (imagens de treino)
> └───val2017/     (imagens de validação)
> ```

1. Coloque o dataset no caminho acima e os **pesos pré-treinados** em `YOLOX/weights/`

```bash
# pesos base do YOLOX-S (usados no fine-tuning via -c)
wget -P YOLOX/weights \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

2. Entre na pasta do YOLOX

```bash
cd YOLOX
```

3. Buildar o container

```bash
docker compose build yolox
```

4. Abrir um shell interativo dentro do container

```bash
docker compose run --rm yolox /bin/bash
```

Os comandos a seguir rodam **dentro** do container (prompt em `/workspace/YOLOX`).
O dataset está em `datasets/vehicles`, os pesos em `weights/` e as saídas vão para
`YOLOX_outputs/` (todos compartilhados com o host).

5. Rodar o treinamento (fine-tuning a partir do `yolox_s.pth`)

```bash
python3 tools/train.py \
  -f exps/example/custom/yolox_s_vehicles.py \
  -d 1 -b 16 --fp16 -o \
  -c weights/yolox_s.pth
```

> Flags: `-d 1` = 1 GPU, `-b 16` = batch size, `--fp16` = precisão mista,
> `-o` = ocupa memória de GPU de forma otimizada, `-c` = checkpoint base.

**Early stopping:** já configurado no Exp (`self.early_stop_patience = 20`) — o treino
para se o **AP50:95 de validação** não melhorar por 20 avaliações seguidas. Coloque `0`
para desativar. O melhor modelo é salvo automaticamente em
`YOLOX_outputs/yolox_s_vehicles/best_ckpt.pth` — use esse na avaliação/inferência.

6. Avaliar um checkpoint treinado

```bash
python3 tools/eval.py \
  -f exps/example/custom/yolox_s_vehicles.py \
  -d 1 -b 16 --fp16 \
  -c YOLOX_outputs/yolox_s_vehicles/best_ckpt.pth
```

> **Alternativa sem shell interativo:** `docker compose up yolox` roda direto o comando
> de treino já configurado no `docker-compose.yml` (precisa dos pesos e do dataset já no lugar).

#### DETR

> **Layout do dataset (DETR):** diferente do YOLOX, o DETR usa o export padrão do
> Roboflow, com as anotações na **raiz** do dataset (não em uma pasta `annotations/`):
> ```
> MobIA 5.4.coco
> ├───train/            (imagens de treino)
> ├───valid/            (imagens de validação)
> ├───train_coco.json   (anotações de treino)
> └───valid_coco.json   (anotações de validação)
> ```

1. Entre na pasta específica do DETR

```bash
cd DETR
```

2. Configure o caminho do dataset

O `docker compose` lê automaticamente um arquivo `.env`. Crie-o a partir do exemplo
e ajuste `DATASET_PATH` para a raiz do seu dataset:

```bash
cp .env_example .env
```

3. Buildar o container específico do DETR

```bash
docker build -t detr-tpf:latest .
```

4. Abrir um shell interativo dentro do container

```bash
docker compose run --rm detr /bin/bash
```

Os comandos a seguir rodam **dentro** do container (prompt em `/workspace/detr`).
O dataset está montado em `/workspace/detr/coco`, e `weights/` e `outputs/` são
compartilhados com o host (pastas `DETR/weights` e `DETR/outputs`).

5. Preparar os pesos pré-treinados para fine-tuning

Baixa o DETR pré-treinado no COCO e remove a cabeça de classificação (91 → 6 classes):

```bash
python prepare_finetune_weights.py            # ResNet-50 → weights/detr-r50_finetune.pth
# ou para o backbone maior:
python prepare_finetune_weights.py --model r101   # → weights/detr-r101_finetune.pth
```

6. Rodar o treinamento (fine-tuning)

```bash
python main.py \
  --coco_path /workspace/detr/coco \
  --output_dir /workspace/detr/outputs \
  --resume weights/detr-r50_finetune.pth \
  --batch_size 2 --epochs 150 --lr_drop 100 --num_workers 4 \
  --early_stop_patience 20
```

> Para o **R101**, use `--resume weights/detr-r101_finetune.pth --backbone resnet101`
> (e considere `--batch_size 1` se faltar memória de GPU).

**Early stopping:** `--early_stop_patience 20` para o treino se o **mAP de validação**
não melhorar por 20 épocas seguidas (use `0` para desativar). O melhor modelo é salvo
em `outputs/checkpoint_best.pth` — use esse arquivo na avaliação, não o `checkpoint.pth`
(que é sempre o da última época). Ajuste fino com `--early_stop_min_delta` (padrão `0.001`).

Os checkpoints e logs ficam em `DETR/outputs/` (no host).

7. Avaliar um checkpoint treinado

```bash
python main.py --eval --no_aux_loss \
  --coco_path /workspace/detr/coco \
  --resume outputs/checkpoint_best.pth --batch_size 2
```

> **Alternativa sem shell interativo:** se preferir, o `docker compose up detr` roda
> direto o comando de treino já configurado no `docker-compose.yml` (passos 5 e 6
> juntos não — os pesos ainda precisam ser gerados uma vez com o passo 5).

#### R-CNN

1. Entre na pasta específica do R-CNN

```bash
cd .\R-CNN\  
```

2. Buildar o container específico do R-CNN
```bash
docker compose build rcnn
```

3. Rodar o treinamento
```bash
docker compose up rcnn
```

#### MASK R-CNN

Implementação [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
(TensorFlow 1.x / Keras), em `M-RCNN/Mask_RCNN/`.

> **Layout do dataset (Mask R-CNN):** mesmo formato COCO do DETR — anotações na
> **raiz** do dataset, com `segmentation` (polígonos ou RLE), pois o Mask R-CNN
> treina máscaras de instância:
> ```
> MobIA 5.4.coco
> ├───train/            (imagens de treino)
> ├───valid/            (imagens de validação)
> ├───train_coco.json   (anotações de treino)
> └───valid_coco.json   (anotações de validação)
> ```

1. Entre na pasta específica do Mask R-CNN

```bash
cd M-RCNN
```

2. Configure o caminho do dataset

O `docker compose` lê automaticamente um arquivo `.env`. Crie-o a partir do exemplo
e ajuste `DATASET_PATH` para a raiz do seu dataset no host:

```bash
cp .env_example .env
```

3. Buildar o container específico do Mask R-CNN

A imagem parte do container **NGC TF1** da NVIDIA (TF 1.15 com CUDA 11), necessário
para rodar TensorFlow 1.x em GPUs Ampere+ (ex.: RTX 3060). Requer Docker +
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose build mrcnn
```

4. Abrir um shell interativo dentro do container

```bash
docker compose run --rm mrcnn /bin/bash
```

Os comandos a seguir rodam **dentro** do container (prompt em `/workspace/Mask_RCNN`).
O dataset está montado em `/workspace/Dataset` (`DATASET_PATH` já aponta para lá),
e o repositório é compartilhado com o host — então os pesos baixados e os
checkpoints em `logs/` ficam salvos na sua máquina.

5. Rodar o treinamento (fine-tuning a partir dos pesos de COCO)

```bash
python samples/custom/custom.py train --weights=coco \
  --epochs 100 --patience 20 --min-delta 0.001 --batch 2
```

> Na primeira vez, `--weights=coco` baixa `mask_rcnn_coco.h5` automaticamente.
> Use `--weights=imagenet` para treinar do zero ou `--weights=last` para continuar
> o último treino. O número de classes é lido automaticamente do `train_coco.json`.

**Early stopping:** ao fim de cada época o script mede o **mAP@0.5:0.95 de validação**
(via pycocotools, mesma métrica do DETR/YOLOX) e para o treino se não melhorar por
`--patience` épocas seguidas (use `0` para desativar; ajuste fino com `--min-delta`).
O melhor modelo é salvo em `logs/<run>/best_custom.h5` — use esse na
avaliação/inferência.

O schedule (épocas, learning rate, camadas treinadas via `--layers heads|all|3+|4+|5+`)
e a configuração (`NUM_CLASSES`, `IMAGES_PER_GPU`, `STEPS_PER_EPOCH`) ficam em
`samples/custom/custom.py`. Reduza `IMAGES_PER_GPU` para 1 se faltar memória de GPU.
Os checkpoints ficam em `M-RCNN/Mask_RCNN/logs/` (no host).

> **Alternativa sem shell interativo:** `docker compose up mrcnn` roda direto o
> comando de treino já configurado no `docker-compose.yml` (com `--weights=coco`).
