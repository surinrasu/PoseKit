import os
import time
import torch
import torch.optim as optim
import numpy as np
import cv2

from lib.utils import maxPoint,extract_keypoints

_range_weight_x = np.array([[x for x in range(48)] for _ in range(48)])
_range_weight_y = _range_weight_x.T

def getSchedu(schedu, optimizer):
    if 'default' in schedu:
        factor = float(schedu.strip().split('-')[1])
        patience = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='max', factor=factor, patience=patience,min_lr=0.000001)
    elif 'step' in schedu:
        step_size = int(schedu.strip().split('-')[1])
        gamma = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif 'SGDR' in schedu:
        T_0 = int(schedu.strip().split('-')[1])
        T_mult = int(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=T_0,
                                                            T_mult=T_mult)
    elif 'MultiStepLR' in schedu:
        milestones = [int(x) for x in schedu.strip().split('-')[1].split(',')]
        gamma = float(schedu.strip().split('-')[2])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=milestones,
                                                gamma=gamma)

    else:
        raise Exception("Unknow schedu.")

    return scheduler

def getOptimizer(optims, model, learning_rate, weight_decay):
    if optims=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optims=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise Exception("Unknow optims.")
    return optimizer

def clipGradient(optimizer, grad_clip=1):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def pkmodelDecode(data, kps_mask=None,mode='output', num_joints = 17,
                img_size=192, hm_th=0.1):

    if mode == 'output':
        batch_size = data[0].size(0)

        heatmaps = data[0].detach().cpu().numpy()

        heatmaps[heatmaps < hm_th] = 0

        centers = data[1].detach().cpu().numpy()

        regs = data[2].detach().cpu().numpy()
        offsets = data[3].detach().cpu().numpy()

        cx,cy = maxPoint(centers)

        dim0 = np.arange(batch_size,dtype=np.int32).reshape(batch_size,1)
        dim1 = np.zeros((batch_size,1),dtype=np.int32)

        res = []
        for n in range(num_joints):

            reg_x_origin = (regs[dim0,dim1+n*2,cy,cx]+0.5).astype(np.int32)
            reg_y_origin = (regs[dim0,dim1+n*2+1,cy,cx]+0.5).astype(np.int32)
            reg_x = reg_x_origin+cx
            reg_y = reg_y_origin+cy

            reg_x = np.reshape(reg_x, (reg_x.shape[0],1,1))
            reg_y = np.reshape(reg_y, (reg_y.shape[0],1,1))
            reg_x = reg_x.repeat(48,1).repeat(48,2)
            reg_y = reg_y.repeat(48,1).repeat(48,2)

            range_weight_x = np.reshape(_range_weight_x,(1,48,48)).repeat(reg_x.shape[0],0)
            range_weight_y = np.reshape(_range_weight_y,(1,48,48)).repeat(reg_x.shape[0],0)
            tmp_reg_x = (range_weight_x-reg_x)**2
            tmp_reg_y = (range_weight_y-reg_y)**2
            tmp_reg = (tmp_reg_x+tmp_reg_y)**0.5+1.8
            tmp_reg = heatmaps[:,n,...]/tmp_reg

            tmp_reg = tmp_reg[:,np.newaxis,:,:]
            reg_x,reg_y = maxPoint(tmp_reg, center=False)

            reg_x[reg_x>47] = 47
            reg_x[reg_x<0] = 0
            reg_y[reg_y>47] = 47
            reg_y[reg_y<0] = 0

            score = heatmaps[dim0,dim1+n,reg_y,reg_x]
            offset_x = offsets[dim0,dim1+n*2,reg_y,reg_x]
            offset_y = offsets[dim0,dim1+n*2+1,reg_y,reg_x]
            res_x = (reg_x+offset_x)/(img_size//4)
            res_y = (reg_y+offset_y)/(img_size//4)

            res_x[score<hm_th] = -1
            res_y[score<hm_th] = -1

            res.extend([res_x, res_y])

        res = np.concatenate(res,axis=1)

    elif mode == 'label':
        kps_mask = kps_mask.detach().cpu().numpy()

        data = data.detach().cpu().numpy()
        batch_size = data.shape[0]

        heatmaps = data[:,:17,:,:]
        centers = data[:,17:18,:,:]
        regs = data[:,18:52,:,:]
        offsets = data[:,52:,:,:]

        cx,cy = maxPoint(centers)
        dim0 = np.arange(batch_size,dtype=np.int32).reshape(batch_size,1)
        dim1 = np.zeros((batch_size,1),dtype=np.int32)

        res = []
        for n in range(num_joints):

            reg_x_origin = (regs[dim0,dim1+n*2,cy,cx]+0.5).astype(np.int32)
            reg_y_origin = (regs[dim0,dim1+n*2+1,cy,cx]+0.5).astype(np.int32)

            reg_x = reg_x_origin+cx
            reg_y = reg_y_origin+cy

            reg_x[reg_x>47] = 47
            reg_x[reg_x<0] = 0
            reg_y[reg_y>47] = 47
            reg_y[reg_y<0] = 0

            offset_x = offsets[dim0,dim1+n*2,reg_y,reg_x]
            offset_y = offsets[dim0,dim1+n*2+1,reg_y,reg_x]
            res_x = (reg_x+offset_x)/(img_size//4)
            res_y = (reg_y+offset_y)/(img_size//4)

            res_x[kps_mask[:,n]==0] = -1
            res_y[kps_mask[:,n]==0] = -1
            res.extend([res_x, res_y])

        res = np.concatenate(res,axis=1)
    return res

import time
import gc
import os
import torch
import torch.nn as nn
import numpy as np

from lib.loss import PoseKitModelLoss
from lib.utils import printDash
from lib.utils import myAcc

class Task():
    def __init__(self, cfg, model):

        self.cfg = cfg

        gpu_id = str(self.cfg.get("GPU_ID", "")).strip()
        if gpu_id.lower() == "mps":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                print("[WARN] MPS requested but not available, using CPU.")
                self.device = torch.device("cpu")
        elif gpu_id != "":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        self.loss_func = PoseKitModelLoss()

        self.optimizer = getOptimizer(self.cfg['optimizer'],
                                    self.model,
                                    self.cfg['learning_rate'],
                                    self.cfg['weight_decay'])

        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

    def train(self, train_loader, val_loader):

        start_epoch = 0
        resume_path = str(self.cfg.get('resume_path', '')).strip()
        if resume_path:
            start_epoch = self.loadCheckpoint(resume_path)

        patience = int(self.cfg.get("early_stop_patience", 0))
        min_delta = float(self.cfg.get("early_stop_min_delta", 0.0))
        best_val_acc = -1.0
        epochs_no_improve = 0

        for epoch in range(start_epoch, self.cfg['epochs']):

            self.onTrainStep(train_loader, epoch)
            val_acc = self.onValidation(val_loader, epoch)

            if patience > 0:
                if val_acc > best_val_acc + min_delta:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("[INFO] Early stopping at epoch %d (best val acc: %.4f)" %
                              (epoch + 1, best_val_acc))
                        break

        self.onTrainEnd()

    def evaluate(self, data_loader):
        self.model.eval()

        right_count = np.zeros(self.cfg['num_classes'], dtype=np.int64)
        total_count = np.zeros(self.cfg['num_classes'], dtype=np.int64)
        acc_kps_th = float(self.cfg.get('acc_kps_th', 0.0))
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):

                if batch_idx%100 == 0:
                    print('Finish ',batch_idx)

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = pkmodelDecode(output, kps_mask,mode='output')
                gt = pkmodelDecode(labels, kps_mask,mode='label')

                acc_correct, acc_total = myAcc(
                    pre,
                    gt,
                    img_size=self.cfg['img_size'],
                    kps_mask=kps_mask,
                    kps_th=acc_kps_th,
                )

                right_count += acc_correct
                total_count += acc_total

        valid = total_count > 0
        acc_value = float(np.mean(right_count[valid] / total_count[valid])) if np.any(valid) else 0.0
        print('[Info] acc: {:.3f}% \n'.format(100. * acc_value))

    def evaluateTest(self, data_loader):
        self.model.eval()

        right_count = np.zeros(self.cfg['num_classes'], dtype=np.int64)
        total_count = np.zeros(self.cfg['num_classes'], dtype=np.int64)
        acc_kps_th = float(self.cfg.get('acc_kps_th', 0.0))
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):

                if batch_idx%100 == 0:
                    print('Finish ',batch_idx)

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs).cpu().numpy()

                pre = []
                for i in range(7):
                    if output[i*3+2]>0.1:
                        pre.extend([output[i*3],output[i*3+1]])
                    else:
                        pre.extend([-1,-1])
                pre = np.array([pre])

                gt = pkmodelDecode(labels, kps_mask,mode='label')
                acc_correct, acc_total = myAcc(
                    pre,
                    gt,
                    img_size=self.cfg['img_size'],
                    kps_mask=kps_mask,
                    kps_th=acc_kps_th,
                )

                right_count += acc_correct
                total_count += acc_total

        valid = total_count > 0
        acc_value = float(np.mean(right_count[valid] / total_count[valid])) if np.any(valid) else 0.0
        print('[Info] acc: {:.3f}% \n'.format(100. * acc_value))

    def onTrainStep(self,train_loader, epoch):

        self.model.train()

        right_count = np.array([0]*self.cfg['num_classes'], dtype=np.int64)
        total_count = np.zeros(self.cfg['num_classes'], dtype=np.int64)
        acc_kps_th = float(self.cfg.get('acc_kps_th', 0.0))

        for batch_idx, (imgs, labels, kps_mask,img_names) in enumerate(train_loader):

            labels = labels.to(self.device)
            imgs = imgs.to(self.device)
            kps_mask = kps_mask.to(self.device)

            output = self.model(imgs)

            heatmap_loss,bone_loss,center_loss,regs_loss,offset_loss = self.loss_func(output, labels, kps_mask)

            total_loss = heatmap_loss+center_loss+regs_loss+offset_loss+bone_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])
            self.optimizer.step()

            pre = pkmodelDecode(output,kps_mask, mode='output')

            gt = pkmodelDecode(labels,kps_mask, mode='label')

            acc_correct, acc_total = myAcc(
                pre,
                gt,
                img_size=self.cfg['img_size'],
                kps_mask=kps_mask,
                kps_th=acc_kps_th,
            )

            right_count += acc_correct
            total_count += acc_total

            avg_acc = float(np.mean(right_count[total_count > 0] / total_count[total_count > 0])) if np.any(total_count > 0) else 0.0
            batch_acc = float(np.mean(acc_correct[acc_total > 0] / acc_total[acc_total > 0])) if np.any(acc_total > 0) else 0.0

            if batch_idx%self.cfg['log_interval']==0:
                print('\r',
                        '%d/%d '
                        '[%d/%d] '
                        'loss: %.4f '
                        '(heatmap: %.3f '
                        'bone: %.3f '
                        'center: %.3f '
                        'reg: %.3f '
                        'offset: %.3f) - '
                        'acc: %.4f' % (epoch+1,self.cfg['epochs'],
                                        batch_idx, len(train_loader.dataset)/self.cfg['batch_size'],
                                        total_loss.item(),
                                        heatmap_loss.item(),
                                        bone_loss.item(),
                                        center_loss.item(),
                        regs_loss.item(),
                        offset_loss.item(),
                        avg_acc),
                        end='',flush=True)
                if self.cfg.get('log_batch_acc', False):
                    print(' (batch: %.4f)         ' % batch_acc, end='', flush=True)
                else:
                    print('         ', end='', flush=True)
            max_batches = int(self.cfg.get('max_train_batches', 0))
            if max_batches and (batch_idx + 1) >= max_batches:
                print('\n[Info] Reached max_train_batches=%d, stopping epoch early.' % max_batches)
                break
        print()

    def onTrainEnd(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()

    def onValidation(self, val_loader, epoch):

        num_test_batches = 0.0
        self.model.eval()

        right_count = np.array([0]*self.cfg['num_classes'], dtype=np.int64)
        total_count = np.zeros(self.cfg['num_classes'], dtype=np.int64)
        acc_kps_th = float(self.cfg.get('acc_kps_th', 0.0))
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(val_loader):
                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                heatmap_loss,bone_loss,center_loss,regs_loss,offset_loss = self.loss_func(output, labels, kps_mask)
                total_loss = heatmap_loss+center_loss+regs_loss+offset_loss+bone_loss

                pre = pkmodelDecode(output, kps_mask,mode='output')
                gt = pkmodelDecode(labels, kps_mask,mode='label')
                acc_correct, acc_total = myAcc(
                    pre,
                    gt,
                    img_size=self.cfg['img_size'],
                    kps_mask=kps_mask,
                    kps_th=acc_kps_th,
                )

                right_count += acc_correct
                total_count += acc_total

            valid = total_count > 0
            acc_value = float(np.mean(right_count[valid] / total_count[valid])) if np.any(valid) else 0.0

            print('LR: %f - '
                ' [Val] loss: %.5f '
                '[heatmap: %.4f '
                'bone: %.4f '
                'center: %.4f '
                'reg: %.4f '
                'offset: %.4f] - '
                'acc: %.4f          ' % (
                                self.optimizer.param_groups[0]["lr"],
                                total_loss.item(),
                                heatmap_loss.item(),
                                bone_loss.item(),
                                center_loss.item(),
                                regs_loss.item(),
                                offset_loss.item(),
                                acc_value),
                                )
            print()

        if 'default' in self.cfg['scheduler']:
            self.scheduler.step(acc_value)
        else:
            self.scheduler.step()

        save_name = 'e%d_valacc%.5f.pth' % (epoch+1, acc_value)
        self.modelSave(save_name)
        if self.cfg.get("save_last", True):
            self.saveCheckpoint("last.pth", epoch, acc_value)
        return acc_value

    def onTest(self):
        self.model.eval()

        res_list = []
        with torch.no_grad():
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):

                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list

    def modelLoad(self,model_path, data_parallel = False):
        self.model.load_state_dict(torch.load(model_path), strict=True)

        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def modelSave(self, save_name):
        torch.save(self.model.state_dict(), os.path.join(self.cfg['save_dir'], save_name))

    def saveCheckpoint(self, save_name, epoch, acc_value=None):
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "acc": acc_value,
            "cfg": self.cfg,
        }
        torch.save(payload, os.path.join(self.cfg['save_dir'], save_name))

    def loadCheckpoint(self, path):
        if not os.path.exists(path):
            print("[WARN] resume_path not found: %s" % path)
            return 0

        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"], strict=bool(self.cfg.get("resume_strict", True)))
            if self.cfg.get("resume_optimizer", True) and "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.cfg.get("resume_scheduler", True) and "scheduler" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            print("[INFO] Resumed checkpoint '%s' at epoch %d" % (path, start_epoch))
            return max(start_epoch, 0)

        # Fallback: treat as plain model weights.
        self.model.load_state_dict(ckpt, strict=bool(self.cfg.get("resume_strict", True)))
        print("[INFO] Loaded model weights from '%s' (no optimizer/scheduler state)" % path)
        return 0
