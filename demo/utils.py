import boto3, os, json
import cv2, glob
import numpy as np
import pandas as pd
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt


def cv2_imshow(img):
    plt.figure(figsize=[15, 10])
    plt.imshow(img[:, :, ::-1])
    plt.show()


def get_iMerit_dicts(data_dir, csv_files, bucket_name="ai-appen-projects"):
    offset = 5
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    dataset_dicts = []
    no_wrong_annot = 0
    for annotation_file in csv_files:
        print(f'annotation_file: {annotation_file}')
        video_name = annotation_file.split('-')[0]
        img_dir = os.path.join(data_dir, video_name)
        tool_db = pd.read_csv(os.path.join(data_dir, annotation_file))

        for i in range(len(tool_db)):
            img_path = tool_db["image_url"][i]
            img_name = img_path.split("/")[-1]
            os.makedirs(img_dir, exist_ok=True)
            dst_img_path = os.path.join(img_dir, img_name)
            s3_subdir = os.path.join(
                img_path.split("/")[-3],
                img_path.split("/")[-2], img_name)
            if not os.path.exists(dst_img_path):
                s3.download_file(bucket_name, s3_subdir, dst_img_path)

            annotations = tool_db["annotation"][i]
            annotations_json = json.loads(annotations)
            if len(annotations_json) == 0:
                continue
            record = dict()
            record["file_name"] = dst_img_path
            image = cv2.imread(dst_img_path)
            height, width = image.shape[:2]
            record["image_id"] = i
            record["height"] = height
            record["width"] = width
            objs = []
            for j in range(len(annotations_json)):
                annotations_dict = annotations_json[j]
                if "Keypoints" != annotations_dict['class']:
                    coord = annotations_dict["coordinates"]
                    if isinstance(coord, list):
                        no_wrong_annot += 1
                        continue
                    obj = {
                        "bbox": [
                            coord["x"] - offset, coord["y"] - offset,
                            (coord["x"] + coord["w"] + offset),
                            (coord["y"] + coord["h"] + offset)
                        ],
                        "bbox_mode":
                        BoxMode.XYXY_ABS,
                        "category_id":
                        0,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    print(f'number of wrong annotation: {no_wrong_annot}')
    return dataset_dicts
