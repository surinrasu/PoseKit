import sys

import torch
import math
import numpy as np

import torch.nn.functional as F
import cv2

_img_size = 192
_feature_map_size = _img_size//4

_center_weight_path = 'lib/data/center_weight_origin.npy'

class JointBoneLoss(torch.nn.Module):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i+1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j
    def forward(self, joint_out, joint_gt):
        J = torch.norm(joint_out[:,self.id_i,:] - joint_out[:,self.id_j,:], p=2, dim=-1, keepdim=False)
        Y = torch.norm(joint_gt[:,self.id_i,:] - joint_gt[:,self.id_j,:], p=2, dim=-1, keepdim=False)
        loss = torch.abs(J-Y)
        loss = torch.sum(loss)/joint_out.shape[0]/len(self.id_i)
        return loss

class PoseKitModelLoss(torch.nn.Module):
    def __init__(self, use_target_weight=False, target_weight=[1]):
        super(PoseKitModelLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight
        self.target_weight=target_weight

        self.center_weight = torch.from_numpy(np.load(_center_weight_path))
        self.make_center_w = False

        self.boneloss = JointBoneLoss(17)

    def l1(self, pre, target,kps_mask):
        return torch.sum(torch.abs(pre - target)*kps_mask)/ (kps_mask.sum() + 1e-4)

    def l2_loss(self, pre, target):
        loss = (pre - target)
        loss = (loss * loss) / 2 / pre.shape[0]

        return loss.sum()

    def centernetfocalLoss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def myMSEwithWeight(self, pre, target):
        loss = torch.pow((pre-target),2)

        weight_mask = target*8+1

        loss = loss*weight_mask

        loss = torch.sum(loss)/target.shape[0]/target.shape[1]

        return loss

    def heatmapL1(self, pre, target):
        loss = torch.abs(pre-target)

        weight_mask = target*4+1

        loss = loss*weight_mask

        loss = torch.sum(loss)/target.shape[0]/target.shape[1]
        return loss

    def boneLoss(self, pred, target):
        #[64, 7, 48, 48]
        def _Frobenius(mat1, mat2):
            return torch.pow(torch.sum(torch.pow(mat1-mat2,2)),0.5)

        _bone_idx = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[2,4]]

        loss = 0
        for bone_id in _bone_idx:
            bone_pre = pred[:,bone_id[0],:,:]-pred[:,bone_id[1],:,:]
            bone_gt = target[:,bone_id[0],:,:]-target[:,bone_id[1],:,:]

            f = _Frobenius(bone_pre,bone_gt)
            loss+=f

        loss = loss/len(_bone_idx)/pred.shape[0]
        return loss

    def bgLoss(self, pre, target):
        ##[64, 7, 48, 48]

        bg_pre = torch.sum(pre, axis=1)
        bg_pre = 1-torch.clamp(bg_pre, 0, 1)

        bg_gt = torch.sum(target, axis=1)
        bg_gt = 1-torch.clamp(bg_gt, 0, 1)

        #weight_mask = (1-bg_gt)*4+1

        loss = torch.sum(torch.pow((bg_pre-bg_gt),2))/pre.shape[0]

        return loss

    def heatmapLoss(self, pred, target, batch_size):
        return self.myMSEwithWeight(pred,target)

    def centerLoss(self, pred, target, batch_size):
        return self.myMSEwithWeight(pred, target)

    def regsLoss(self, pred, target, cx0, cy0,  kps_mask, batch_size, num_joints):
        _dim0 = torch.arange(0,batch_size).long()
        _dim1 = torch.zeros(batch_size).long()

        loss = 0
        for idx in range(num_joints):

            gt_x = target[_dim0,_dim1+idx*2,cy0,cx0]
            gt_y = target[_dim0,_dim1+idx*2+1,cy0,cx0]

            pre_x = pred[_dim0,_dim1+idx*2,cy0,cx0]
            pre_y = pred[_dim0,_dim1+idx*2+1,cy0,cx0]

            loss+=self.l1(gt_x,pre_x,kps_mask[:,idx])
            loss+=self.l1(gt_y,pre_y,kps_mask[:,idx])

        return loss / num_joints

    def offsetLoss(self, pred, target,  cx0, cy0, regs, kps_mask, batch_size, num_joints):
        _dim0 = torch.arange(0,batch_size).long()
        _dim1 = torch.zeros(batch_size).long()
        loss = 0
        for idx in range(num_joints):
            gt_x = regs[_dim0,_dim1+idx*2,cy0,cx0].long()+cx0
            gt_y = regs[_dim0,_dim1+idx*2+1,cy0,cx0].long()+cy0

            gt_x[gt_x>47]=47
            gt_x[gt_x<0]=0
            gt_y[gt_y>47]=47
            gt_y[gt_y<0]=0

            gt_offset_x = target[_dim0,_dim1+idx*2,gt_y,gt_x]
            gt_offset_y = target[_dim0,_dim1+idx*2+1,gt_y,gt_x]

            pre_offset_x = pred[_dim0,_dim1+idx*2,gt_y,gt_x]
            pre_offset_y = pred[_dim0,_dim1+idx*2+1,gt_y,gt_x]

            loss+=self.l1(gt_offset_x,pre_offset_x,kps_mask[:,idx])
            loss+=self.l1(gt_offset_y,pre_offset_y,kps_mask[:,idx])

        return loss / num_joints

    def maxPointPth(self, heatmap, center=True):
        if center:
            heatmap = heatmap*self.center_weight[:heatmap.shape[0],...]

        n,c,h,w = heatmap.shape
        heatmap = heatmap.reshape((n, -1)) #64, 48x48
        max_v,max_id = torch.max(heatmap, 1)#64, 1

        y = max_id//w
        x = max_id%w

        return x,y

    def forward(self, output, target, kps_mask):
        batch_size = output[0].size(0)
        num_joints = output[0].size(1)

        heatmaps = target[:,:17,:,:]
        centers = target[:,17:18,:,:]
        regs = target[:,18:52,:,:]
        offsets = target[:,52:,:,:]

        heatmap_loss = self.heatmapLoss(output[0], heatmaps, batch_size)

        bone_loss = self.boneLoss(output[0], heatmaps)
        #print(heatmap_loss)
        center_loss = self.centerLoss(output[1], centers, batch_size)

        if not self.make_center_w:
            self.center_weight = torch.reshape(self.center_weight,(1,1,48,48))
            self.center_weight = self.center_weight.repeat((output[1].shape[0],output[1].shape[1],1,1))

            self.center_weight = self.center_weight.to(target.device)
            self.make_center_w = True
            self.center_weight.requires_grad_(False)

        cx0, cy0 = self.maxPointPth(centers)
        cx0 = torch.clip(cx0,0,_feature_map_size-1).long()
        cy0 = torch.clip(cy0,0,_feature_map_size-1).long()

        regs_loss = self.regsLoss(output[2], regs, cx0, cy0, kps_mask,batch_size, num_joints)
        offset_loss = self.offsetLoss(output[3], offsets,
                            cx0, cy0,regs,
                            kps_mask,batch_size, num_joints)

        return [heatmap_loss,bone_loss,center_loss,regs_loss,offset_loss]

pkmodelLoss = PoseKitModelLoss(use_target_weight=False)

def calculate_loss(predict, label):
    loss = pkmodelLoss(predict, label)
    return loss
