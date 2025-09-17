import glob
import os
import random
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET


def load_images_and_labels(img_dir, label_dir, label2idx):
    img_infos = []
    for label_file in tqdm(glob.glob(os.path.join(label_dir, '*.xml'))):
        im_info = {}
        im_info['img_id'] = os.path.basename(label_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
        label_info = ET.parse(label_file)
        root = label_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        img_info['width'] = width
        img_info['height'] = height
        detections = []
        
        for obj in label_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        img_info['detections'] = detections
        img_infos.append(img_info)
    print('Total {} images found'.format(len(img_infos)))
    return img_infos


class VOCDataset(Dataset):
    def __init__(self, split, img_dir, label_dir):
        self.split = split
        self.img_dir = img_dir
        self.label_dir = label_dir
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(img_dir, label_dir, self.label2idx)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = Image.open(img_info['filename'])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_tensor = torchvision.transforms.ToTensor()(img)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in img_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in img_info['detections']])
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return img_tensor, targets, img_info['filename']
