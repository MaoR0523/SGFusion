import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class FusionLoss(nn.Module):
    def __init__(self,background_classes=[1,2],suppress_classes=[9,10,11]):
        super(FusionLoss, self).__init__()
        self.sobelconv=Sobelxy()    
        self.bn = nn.BatchNorm2d(1)
        
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.register_buffer('background_classes', 
                            torch.tensor(background_classes, dtype=torch.long))
        self.register_buffer('suppress_classes', 
                            torch.tensor(suppress_classes, dtype=torch.long))  
    def create_mask(self, map, ir):
        """
        生成有效区域掩码
        输入: 
            map -> (B, H, W) 分割结果
            ir  -> (B, 1, H, W) 红外图像
        输出: 
            mask -> (B, 1, H, W)
        """
        # 1. 处理背景掩码 (B, 1, H, W)
        # 建议直接在 map 上操作，无需 unsqueeze(-1) 和 permute
        background_mask = torch.isin(map, self.background_classes).unsqueeze(1).float()

        # 2. 生成需要抑制的类别区域布尔掩码 (B, 1, H, W)
        tmp_mask = torch.isin(map, self.suppress_classes).unsqueeze(1)
        tmp_mask_float = tmp_mask.float()

        # 3. 按图片计算独立阈值 (B, 1, 1, 1)
        # 计算每张图中指定区域的红外亮度总和
        sum_ir = torch.sum(ir * tmp_mask_float, dim=(2, 3), keepdim=True)
        # 计算每张图中指定区域的像素个数
        count_pixel = torch.sum(tmp_mask_float, dim=(2, 3), keepdim=True)

        # 避免除以 0 (防止某张图完全没有 suppress_classes)
        threshold = sum_ir / (count_pixel + 1e-6)

        # 4. 生成抑制掩码 (B, 1, H, W)
        # 只有在 tmp_mask 区域内且大于该图阈值的像素才为 True
        suppress_mask = (ir * tmp_mask_float > threshold).float()

        # 返回合并后的掩码 (确保是 float 类型)
        return background_mask + suppress_mask

        
    def forward(self,image_vis,image_ir,generate_img,labels):
        image_vis=image_vis*0.5+0.5
        image_ir=image_ir*0.5+0.5
        valid_mask = self.create_mask(labels,image_ir)

        image_y=image_vis[:,:1,:,:]
        image_ir1=(1-valid_mask)*image_ir
        x_in_max=torch.max(image_y,image_ir1)
        loss_in=F.l1_loss(x_in_max,generate_img)        
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        aux = torch.abs(x_in_max-generate_img)
        
        aux_weight = torch.tanh(self.bn(aux))+1
        
        
        
        return loss_in,loss_grad,torch.squeeze(aux_weight, dim=1)
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.bn = nn.BatchNorm2d(1)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class DynamicCELoss(nn.Module):
    def __init__(self, threshold=0.15,ignore_lb=255,eps=1e-7,label_smoothing=0.1):
        super(DynamicCELoss, self).__init__()
        self.num_classes = 15
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.thresh = torch.tensor(threshold, dtype=torch.float)
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(reduction='none')
        self.eps = eps
    def forward(self, logits, labels,aux_weight,flag):
        B,H,W=labels.shape
        real_ce = self.ce(logits, labels)
        # aux_ce = real_ce
        aux_ce = torch.mean(aux_weight*self.criteria(logits, labels))
        # aux_ce = self.dice(logits, labels)
        hard_rate= 0.0
        return real_ce,aux_ce,0.0

class DiceLoss(nn.Module):
    def __init__(self, n_classes = 15, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, C, H, W] 网络的原始输出
        target: [B, H, W] 标签，值在 [0, C-1] 之间
        """
        # 1. 转换 logits 为概率分布
        probs = F.softmax(logits, dim=1)
        
        # 2. 将 target 转换为 one-hot 编码: [B, H, W] -> [B, C, H, W]
        target_one_hot = F.one_hot(target, num_classes=self.n_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # 3. 计算 Intersection 和 Union
        # 在 [B, H, W] 维度上求和，保留 [C] 维度
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)

        # 4. 计算 Dice 系数
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 5. 返回 Dice Loss (通常用 1 - Dice)
        return 1. - dice_score.mean()
        
class Loss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(Loss, self).__init__()
        self.fusionloss = FusionLoss()
        self.segloss = DynamicCELoss() 

    def rgb_to_grayscale(self,tensor):
    
        r = tensor[:, 0:1, :, :]  # 红色通道 (B, 1, H, W)
        g = tensor[:, 1:2, :, :]  # 绿色通道 (B, 1, H, W)
        b = tensor[:, 2:3, :, :]  # 蓝色通道 (B, 1, H, W)
        
        # 使用标准转换公式
        # 灰度 = 0.299 * R + 0.587 * G + 0.114 * B
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray 
    def forward(self,image_vis, image_ir,generate_img,seg,labels,flag):
        image_vis = self.rgb_to_grayscale(image_vis)
        image_ir = self.rgb_to_grayscale(image_ir)
        loss_in,loss_grad,aux_weight = self.fusionloss(image_vis, image_ir,generate_img,labels)
        seg_loss,real_ce,hard_rate= self.segloss(seg, labels,aux_weight,flag)
        return loss_in,loss_grad,seg_loss,real_ce,hard_rate

