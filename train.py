import torch
import torch.nn as nn
import os
import KGnet
import numpy as np
import transforms
import argparse
import cv2
from loss import DetectionLossAll
import config as cfg
import seg_loss
from tqdm import tqdm
from collater import collater
from dataset_kaggle import Kaggle
from dataset_plant import Plant
from dataset_neural import Neural
from dataset_sanmed import Sanmed
from dataset_sanmed_dapi import Sanmed_dapi

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="InstanceHeat")
    parser.add_argument("--data_dir", help="data directory", default="../../../Datasets/kaggle/", type=str)
    parser.add_argument("--resume", help="resume file", default="end_model.pth", type=str)
    parser.add_argument('--input_h', type=int, default=512, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument("--workers", help="workers number", default=4, type=int)
    parser.add_argument("--batch_size", help="batch size", default=2, type=int)
    parser.add_argument("--epochs", help="epochs", default=30, type=int)
    parser.add_argument("--start_epoch", help="start_epoch", default=0, type=int)
    parser.add_argument("--lr", help="learning_rate", default=0.0001, type=int)
    parser.add_argument("--data_parallel", help="data parallel", default=False, type=bool)
    parser.add_argument("--dataset", help="training dataset", default='kaggle', type=str)
    args = parser.parse_args()
    return args



class InstanceHeat(object):
    def __init__(self):
        self.model = KGnet.resnet50(pretrained=True)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.dataset = {'kaggle': Kaggle, 'plant': Plant, 'neural': Neural, 'sanmed': Sanmed, 'sanmed_dapi':Sanmed_dapi}

        # os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
        # model = KGnet.resnet50(pretrained=True)
        # if torch.cuda.device_count()>1:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.dataset = {'kaggle': Kaggle, 'plant': Plant, 'neural': Neural}
        #     self.model = nn.DataParallel(model)
        # else:
        #     self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        #     self.dataset = {'kaggle': Kaggle, 'plant': Plant, 'neural': Neural}
        #     self.model = model.to(self.device)


    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def load_weights(self, resume, dataset):
        self.model.load_state_dict(torch.load(os.path.join('weights_' + dataset, resume)))

    def map_mask_to_image(self, mask, img, color):
        # color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def show_heat_mask(self, mask):
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        return heatmap

    def train(self, args):
        weights_path = os.path.join("weights_"+args.dataset)
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        elif os.path.exists(os.path.join(weights_path,'end_model.pth')):
            self.load_weights(resume=args.resume, dataset=args.dataset)

        self.model = self.model.to(self.device)

        self.model.train()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)

        loss_dec = DetectionLossAll(kp_radius=cfg.KP_RADIUS)
        loss_seg = seg_loss.SEG_loss(height=args.input_h, width=args.input_w)

        data_trans = {'train': transforms.Compose([transforms.ConvertImgFloat(),
                                                   transforms.PhotometricDistort(),
                                                   transforms.Expand(max_scale=2, mean=(0, 0, 0)),
                                                   transforms.RandomMirror_w(),
                                                   transforms.RandomMirror_h(),
                                                   transforms.Resize(h=args.input_h, w=args.input_w)]),

                      'val': transforms.Compose([transforms.ConvertImgFloat(),
                                                 transforms.Resize(h=args.input_h, w=args.input_w)])}

        dataset_module = self.dataset[args.dataset]
        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   transform=data_trans[x])
                 for x in ['train', 'val']}

        # for i in range(100):
        #     show_ground_truth.show_input(dsets.__getitem__(i))


        train_loader = torch.utils.data.DataLoader(dsets['train'],
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   shuffle=True,
                                                   collate_fn=collater)


        val_loader = torch.utils.data.DataLoader(dsets['val'],
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 shuffle=False,
                                                 collate_fn=collater)


        train_loss_dict = []
        val_loss_dict = []
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            print('Epoch {}/{}'.format(epoch, args.start_epoch+args.epochs - 1))
            print('-' * 10)

            train_epoch_loss = self.training(train_loader,loss_dec,loss_seg,optimizer,epoch, dsets['train'])
            train_loss_dict.append(train_epoch_loss)
            scheduler.step(epoch)

            val_epoch_loss = self.validating(val_loader,loss_dec,loss_seg, epoch, dsets['val'])
            val_loss_dict.append(val_epoch_loss)

            np.savetxt('train_loss_{}.txt'.format(args.dataset), train_loss_dict, fmt='%.6f')
            np.savetxt('val_loss_{}.txt'.format(args.dataset), val_loss_dict, fmt='%.6f')

            if epoch % 5 == 0 and epoch-args.start_epoch >0:
                torch.save(self.model.state_dict(), os.path.join(weights_path, '{:d}_{:.4f}_model.pth'.format(epoch, train_epoch_loss)))
            torch.save(self.model.state_dict(), os.path.join(weights_path, 'end_model.pth'))

    def training(self, train_loader, loss_dec, loss_seg, optimizer, epoch, dsets):
        self.model.train()
        running_loss = 0.0

        for data in tqdm(train_loader):
            img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0  = data
            img = img.to(self.device)
            gt_c0 = gt_c0.to(self.device)
            gt_c1 = gt_c1.to(self.device)
            gt_c2 = gt_c2.to(self.device)
            gt_c3 = gt_c3.to(self.device)

            optimizer.zero_grad()

            with torch.enable_grad():
                pr_c0, pr_c1, pr_c2, pr_c3, predictions = self.model(img, bboxes_c0)
                loss1 = loss_dec(pr_c0, gt_c0)+loss_dec(pr_c1, gt_c1)+loss_dec(pr_c2, gt_c2)+loss_dec(pr_c3, gt_c3)
                loss2 = loss_seg(predictions, instance_masks, bboxes_c0)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(dsets)
        print('{} Loss: {:.6}'.format(epoch, epoch_loss))
        return epoch_loss

    def validating(self, val_loader, loss_dec, loss_seg, epoch, dsets):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                img, gt_c0, gt_c1, gt_c2, gt_c3, instance_masks, bboxes_c0  = data
                img = img.to(self.device)
                gt_c0 = gt_c0.to(self.device)
                gt_c1 = gt_c1.to(self.device)
                gt_c2 = gt_c2.to(self.device)
                gt_c3 = gt_c3.to(self.device)
                pr_c0, pr_c1, pr_c2, pr_c3, predictions = self.model(img, bboxes_c0)
                loss1 = loss_dec(pr_c0, gt_c0)+loss_dec(pr_c1, gt_c1)+loss_dec(pr_c2, gt_c2)+loss_dec(pr_c3, gt_c3)
                loss2 = loss_seg(predictions, instance_masks, bboxes_c0)
                loss = loss1 + loss2
                running_loss += loss.item()
        epoch_loss = running_loss / len(dsets)
        print('Valid {} Loss: {:.6}'.format(epoch, epoch_loss))
        return epoch_loss



if __name__ == '__main__':
    args = parse_args()
    args.data_dir = r'/home/xing/Share/Projects/Sanmed/cell_seg/20201127/Dapi_output'
    args.dataset = 'sanmed_dapi'
    args.batch_size = 4
    args.workers = 4
    args.start_epoch = 0
    print(args)
    object_is = InstanceHeat()
    object_is.train(args)
