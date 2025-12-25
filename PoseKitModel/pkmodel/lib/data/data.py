import os
import random
import numpy as np
import cv2
import json
import copy
from torchvision import transforms

from lib.data.data_tools import getDataLoader,getFileNames
from lib.task import pkmodelDecode

class Data():
    def __init__(self, cfg):

        self.cfg = cfg

    def dataBalance(self, data_list):
        new_data_list = copy.deepcopy(data_list)

        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for item in data_list:
            keypoints = item['keypoints']

            kpt_np = np.array(keypoints).reshape((-1,3))
            kpt_np_valid = kpt_np[kpt_np[:,2]>0]

            w = np.max(kpt_np_valid[:,0])-np.min(kpt_np_valid[:,0])
            h = np.max(kpt_np_valid[:,1])-np.min(kpt_np_valid[:,1])

            if (kpt_np[1][1]>kpt_np[0][1] and kpt_np[1][1]>kpt_np[2][1]) or \
                (kpt_np[5][1]>kpt_np[6][1] and kpt_np[5][1]>kpt_np[4][1]):
                count1+=1
                print(item)
                for i in range(2):
                    new_data_list.append(item)

            if kpt_np[2][0]-kpt_np[4][0] < \
                max(kpt_np[1][1]-kpt_np[2][1],kpt_np[5][1]-kpt_np[4][1]):
                count2+=1

                for i in range(2):
                    new_data_list.append(item)

            if (kpt_np[1][1]<kpt_np[2][1]) or \
                (kpt_np[5][1]<kpt_np[4][1]):
                count3+=1

                for i in range(5):
                    new_data_list.append(item)

            if h<w:
                count4+=1
                for i in range(3):
                    new_data_list.append(item)

        print(count1, count2, count3, count4)

        random.shuffle(new_data_list)
        return new_data_list

    def getTrainValDataloader(self):

        with open(self.cfg['train_label_path'],'r') as f:
            train_label_list = json.loads(f.readlines()[0])
            random.shuffle(train_label_list)

        with open(self.cfg['val_label_path'], 'r') as f:
            val_label_list = json.loads(f.readlines()[0])
            random.shuffle(val_label_list)

        print("[INFO] Total train images: %d, val images: %d" %
                (len(train_label_list), len(val_label_list)))

        if self.cfg['balance_data']:
            train_label_list = self.dataBalance(train_label_list)
            val_label_list = self.dataBalance(val_label_list)
            print("[INFO] After balance data, Total train images: %d, val images: %d" %
                    (len(train_label_list), len(val_label_list)))
        else:
            print("[INFO] Not do data balance.")

        input_data = [train_label_list, val_label_list]
        train_loader, val_loader = getDataLoader("trainval",
                                                input_data,
                                                self.cfg)
        return train_loader, val_loader

    def getEvalDataloader(self):
        with open(self.cfg['eval_label_path'], 'r') as f:
            data_label_list = json.loads(f.readlines()[0])

        print("[INFO] Total images: ", len(data_label_list))

        input_data = [data_label_list]
        data_loader = getDataLoader("eval",
                                        input_data,
                                        self.cfg)
        return data_loader

    def getTestDataloader(self):
        data_names = getFileNames(self.cfg['test_img_path'])
        test_loader = getDataLoader("test",
                                    data_names,
                                    self.cfg)
        return test_loader

    def showData(self, data_loader, show_num = 400):
        show_dir = "show_img"
        show_path = os.path.join(self.cfg['save_dir'], show_dir)
        if not os.path.exists(show_path):
            os.makedirs(show_path)

        show_count = 0
        for (imgs, labels_pth, mask, img_names) in data_loader:

            imgs = imgs.cpu().numpy()
            labels = labels_pth.cpu().numpy()

            for i in range(imgs.shape[0]):

                img = np.transpose(imgs[i],axes=[1,2,0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                basename = os.path.basename(img_names[i])[:-4]

                heatmaps = labels[i,:17,:,:]
                centers = labels[i,17:18,:,:]
                regs = labels[i,18:52,:,:]
                offsets = labels[i,52:,:,:]

                cv2.imwrite(os.path.join(show_path,basename+"_origin.jpg"), img)
                cv2.imwrite(os.path.join(show_path,basename+"_centers.jpg"), cv2.resize(centers[0]*255, (192,192)))
                heatmaps = np.sum(heatmaps,axis=0)
                heatmaps = cv2.resize(heatmaps, (192,192))*255
                cv2.imwrite(os.path.join(show_path,basename+"_regs0.jpg"), np.abs(regs[0])*255)

                h,w = img.shape[:2]

                this_label = labels_pth[i]
                this_label = this_label[np.newaxis,...]
                gt = pkmodelDecode(this_label, mask[i].reshape(1,-1),mode='label')
                for i in range(len(gt[0])//2):
                    x = int(gt[0][i*2]*w)
                    y = int(gt[0][i*2+1]*h)
                    if x>0 and y>0:
                        cv2.circle(img, (x, y), 5, (0,255,0), 3)
                cv2.imwrite(os.path.join(show_path,basename+"_gt.jpg"), heatmaps)
                cv2.imwrite(os.path.join(show_path,basename+"_gtimg.jpg"), img)

            show_count += imgs.shape[0]
            if show_count>=show_num:
                break
