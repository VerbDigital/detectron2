import boto3, os, json
import cv2, glob
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from detectron2.structures import BoxMode
import matplotlib.pyplot as plt


def cv2_imshow(img):
    plt.figure(figsize=[15, 10])
    plt.imshow(img[:, :, ::-1])
    plt.show()

def convert_xml_annot(annot_path):
    objs = []
    tree = ET.parse(annot_path)
    root = tree.getroot()
    for idx,log_element in enumerate(root.findall('object/bndbox')):
        xmin = int(log_element.findall('xmin')[0].text)
        xmax = int(log_element.findall('xmax')[0].text)
        ymin = int(log_element.findall('ymin')[0].text)
        ymax = int(log_element.findall('ymax')[0].text)
        # bbox.append([xmin, ymin, xmax, ymax])
        obj = {
            "bbox": [
                xmin, ymin,
                xmax,ymax],
            "bbox_mode":
                BoxMode.XYXY_ABS,
            "category_id":
                0,
        }
        objs.append(obj)
    return objs


def get_m2cai16_dict(data_dir, image_list_filename):
    image_list_path = os.path.join(data_dir, 'ImageSets/Main/', image_list_filename)
    with open(image_list_path, 'r') as f:
        img_list = f.readlines()
    img_list = [x.strip() for x in img_list]
    dataset_dicts = []
    for ind in range(len(img_list)):
        record = dict()
        img_path = os.path.join(data_dir, 'JPEGImages/', img_list[ind] + '.jpg')
        image = cv2.imread(img_path)
        annot_path = os.path.join(data_dir, 'Annotations/', img_list[ind] + '.xml')
        bboxs = convert_xml_annot(annot_path)
        record["file_name"] = img_path
        record["annotations"] = bboxs
        height, width = image.shape[:2]
        record["image_id"] = ind
        record["height"] = height
        record["width"] = width
        dataset_dicts.append(record)
    return dataset_dicts



def create_dataset(iMrit_data_dir, iMerit_csv_files, m2cai_data_dir, m2cai_image_list_path):
    iMerit_dataset = get_iMerit_dicts(iMrit_data_dir, iMerit_csv_files)
    m2cai_dataset = get_m2cai16_dict(m2cai_data_dir, m2cai_image_list_path)
    dataset_dicts = []
    dataset_dicts.extend(iMerit_dataset)
    dataset_dicts.extend(m2cai_dataset)
    print(f'm2cai: {len(m2cai_dataset)}, iMerit_dataset: {len(iMerit_dataset)}, total: {len(dataset_dicts)}')
    return dataset_dicts

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

if __name__ == '__main__':
    data_dir = '/home/mona/share/data/m2cai16-tool-locations/'
    image_list_path = '/home/mona/share/data/m2cai16-tool-locations/ImageSets/Main/val.txt'
    datasets = get_m2cai16_dict(data_dir, image_list_path)
    record = datasets[10]
    img = cv2.imread(record['file_name'])
    objs =  record["annotations"]
    for i in range(len(objs)):
        obj = objs[i]['bbox']
        img = cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255, 0, 0), 2)
    plt.imshow(img[:,:,::-1])
    plt.show()


