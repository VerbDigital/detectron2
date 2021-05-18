import os, torch, glob, random
from detectron2.config import get_cfg
from utils import *
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

surgical_tool_path = "/home/mona/share/data/score/annotation/appen_new_annot"

csv_files = glob.glob(surgical_tool_path + '/*.csv')
N = len(csv_files)
val_inds = np.random.choice(N, N // 4)
train_inds = np.setdiff1d(np.arange(N), val_inds)
train_val_split = {
    'train': [csv_files[i] for i in train_inds],
    'val': [csv_files[i] for i in val_inds]
}
train_dataset_dicts = get_iMerit_dicts(surgical_tool_path,
                                       train_val_split['train'])
val_dataset_dicts = get_iMerit_dicts(surgical_tool_path,
                                     train_val_split['val'])
print(
    f'len-train: {len(train_dataset_dicts)}, len_val: {len(val_dataset_dicts)}'
)

for d in ["train", "val"]:
    dataset_name = "tool_" + d
    DatasetCatalog.register(
        dataset_name,
        lambda d=d: get_iMerit_dicts(surgical_tool_path, train_val_split[d]))
    MetadataCatalog.get(dataset_name).set(thing_classes=["tool"])

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
cfg.DATASETS.TRAIN = ('tool_train', )
cfg.DATASETS.TEST = ('tool_val', )
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.MAX_ITER = 50  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg['MODEL'].keys()
print(f'output dir: {cfg.OUTPUT_DIR}')

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
predictor = DefaultPredictor(cfg)
trainer = DefaultTrainer(cfg)

# with torch.no_grad():
#     evaluator = COCOEvaluator("tool_val", ("bbox",), False, output_dir="./output/")
#     val_loader = build_detection_test_loader(cfg, "tool_val")
#     print(inference_on_dataset(trainer.model, val_loader, evaluator))

print('show prediction')
tool_metadata = MetadataCatalog.get("tool_train")
dataset_dicts = get_iMerit_dicts(surgical_tool_path,
                                 csv_files=["1461-f1763033.csv"])
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im[:, :, ::-1],
        metadata=tool_metadata,
        scale=0.5,
        instance_mode=ColorMode.
        IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

# another equivalent way to evaluate the model is to use `trainer.test`
