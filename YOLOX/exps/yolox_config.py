import os 
import cv2
from yolox.exp import Exp as MyExp
from yolox.data import COCODataset

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        self.exp_name = "treinamento_teste_mobia_yolox"
        self.num_classes = 6

        self.data_dir = "/workspace/YOLOX/datasets"
        self.train_ann = "train_annotations.coco.json"
        self.val_ann = "valid_annotations.coco.json"

        # Configurações de Épocas
        self.max_epoch = 15
        self.no_aug_epochs = 15
        self.print_interval = 10
        self.eval_interval = 10

        self.data_num_workers = 4

    def get_dataset(self, cache=False, cache_type="ram"):
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train", # Sua pasta real de treino
            img_size=self.input_size,
            preproc=None,
            cache=cache,
            cache_type=cache_type
        )
        
        dataset.load_image = lambda index: self._safe_load_image(dataset, index, "train")
        return dataset
    
    def get_eval_dataset(self, **kwargs):
        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="valid", 
            img_size=self.test_size,
            preproc=None
        )
        
        dataset.load_image = lambda index: self._safe_load_image(dataset, index, "valid")
        return dataset

    def _safe_load_image(self, dataset_obj, index, folder_name):
        """
        Função auxiliar que tenta ler a imagem no caminho do JSON.
        Se falhar, força a busca na pasta correta mapeada pelo Docker.
        """
        id_ = dataset_obj.ids[index]
        file_name = dataset_obj.coco.loadImgs(id_)[0]["file_name"]
        
        img_file = os.path.join(dataset_obj.data_dir, dataset_obj.name, file_name)
        img = cv2.imread(img_file)
        
        if img is None:
            img_file = os.path.join(self.data_dir, folder_name, file_name)
            img = cv2.imread(img_file)
            
        assert img is not None, f"Imagem absolutamente nao encontrada em: {img_file}"
        return img