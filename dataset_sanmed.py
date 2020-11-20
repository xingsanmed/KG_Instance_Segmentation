from dataset_base import BaseDataset
import os
import numpy as np
import cv2
import glob
import torch
import torch.utils.data

import transforms
import collater


class Sanmed(BaseDataset):
    def __init__(self, data_dir, phase, transform=None):
        super(Sanmed, self).__init__(data_dir, phase, transform)
        self.class_name = ['__background__', 'sanmed']
        self.num_classes = len(self.class_name)-1

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.img_dir, img_id, "images", img_id+'.jpg')
        img = cv2.imread(imgFile)
        return img

    def load_gt_masks(self, annopath):
        masks = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*" + '.png'))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                masks.append(np.where(mask > 0, 1., 0.))
        return np.asarray(masks, np.float32)


    def load_gt_bboxes(self, annopath):
        bboxes = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*" + '.png'))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([y1, x1, y2, x2])
        return np.asarray(bboxes, np.float32)

    def load_annoFolder(self, index):
        img_id = self.img_ids[index]
        return os.path.join(self.img_dir, img_id, "masks")

    def load_annotation(self, index, type='mask'):
        annoFolder = self.load_annoFolder(index)
        if type=='mask':
            return self.load_gt_masks(annoFolder)
        else:
            return self.load_gt_bboxes(annoFolder)


if __name__ == '__main__':
    # dataset = {'kaggle': Kaggle, 'plant': Plant, 'neural': Neural}
    dataset_module = Sanmed

    data_dir = r'/home/xing/Share/Projects/Sanmed/cell_seg/Data/20201020'
    data_trans = {'train': transforms.Compose([transforms.ConvertImgFloat(),
                                               transforms.PhotometricDistort(),
                                               transforms.Expand(max_scale=2, mean=(0, 0, 0)),
                                               transforms.RandomMirror_w(),
                                               transforms.RandomMirror_h(),
                                               transforms.Resize(h=2048, w=2048)]),

                  'val': transforms.Compose([transforms.ConvertImgFloat(),
                                             transforms.Resize(h=2048, w=2048)])}

    dsets = {x: dataset_module(data_dir= data_dir,
                               phase=x,
                               transform=data_trans[x])
             for x in ['train', 'val']}

    # for i in range(100):
    #     show_ground_truth.show_input(dsets.__getitem__(i))

    train_loader = torch.utils.data.DataLoader(dsets['train'],
                                               batch_size=2,
                                               num_workers=4,
                                               pin_memory=True,
                                               shuffle=True,
                                               collate_fn=collater)

    val_loader = torch.utils.data.DataLoader(dsets['val'],
                                             batch_size=1,
                                             num_workers=4,
                                             pin_memory=True,
                                             shuffle=False,
                                             collate_fn=collater)

    from tqdm import tqdm
    for data in val_loader:
        img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0 = data
