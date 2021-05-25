import torch, torchvision

print(torch.__version__, torch.cuda.is_available())
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.engine import DefaultTrainer
import numpy as np
from utils import *
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy

numpy.version.version
import os
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"

show_example = False
surgical_tool_path = "/home/mona/share/data/score/annotation/appen_new_annot"
import glob

csv_files = glob.glob(surgical_tool_path + '/*.csv')
N = len(csv_files)
val_inds = np.random.choice(N, N // 3)
train_inds = np.setdiff1d(np.arange(N), val_inds)
train_val_split = {
    'train': [csv_files[i] for i in train_inds],
    'val': [csv_files[i] for i in val_inds]
}
m2cai_data_dir = '/home/mona/share/data/m2cai16-tool-locations'

for d in ["train", "val"]:
    dataset_name = "tool_" + d
    DatasetCatalog.register(
        dataset_name,
        lambda d=d: create_dataset(surgical_tool_path, train_val_split[d], m2cai_data_dir, f'{d}.txt'))
    MetadataCatalog.get(dataset_name).set(thing_classes=["tool"])

if show_example:
    dataset_dicts = get_iMerit_dicts(surgical_tool_path,
                                     csv_files=["1461-f1763033.csv"])
    print(len(dataset_dicts), dataset_dicts[0].keys())
    tool_metadata = MetadataCatalog.get("tool_train_1461")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=tool_metadata,
                                scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2_imshow(out.get_image()[:, :, ::-1])

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
cfg.DATASETS.TRAIN = ('tool_train', )
cfg.DATASETS.TEST = ('tool_val', )
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 50  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg['MODEL'].keys()
print(f'output dir: {cfg.OUTPUT_DIR}')

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print('Start evaluation')
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
predictor = DefaultPredictor(cfg)
