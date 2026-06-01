import os 
from yolox.exp import Exp as MyExp
from yolox.data import COCODataset

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        self.exp_name = "treinamento_teste_mobia_yolox"

        self.num_classes = 6

        self.data_dir = "/workspace/YOLOX/datasets"
        self.train_ann = "train_annotations.json"
        self.val_ann = "val_annotations.json"

        self.max_epoch = 15
        self.no_aug_epochs = 15
        self.print_interval = 10
        self.eval_internal = 10

    def get_dataset(self, cache=False, cache_type="ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=None,
            cache=cache,
            cache_type=cache_type
        )
    
    def get_eval_dataset(self, **kwargs):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="valid",
            img_size=self.test_size,
            preproc=None
        )