import torch
import torch.nn as nn
import torch.nn.functional as F

class HighIntensityCNN(nn.Module):
    def __init__(self, in_channels=3):
        """
        高亮度特征提取CNN
        输入: (B, 3, H, W) 红外图像
        输出: (B, 1, H, W) 高亮度特征图
        """
        super().__init__()
        
        # 特征提取主干 (保持空间分辨率)
        self.feature_extractor = nn.Sequential(
            # 第一层: 增强对比度
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            
            
        )
        
        # 高亮度响应层
        self.intensity_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # 输出单通道特征图
            nn.Sigmoid()  # 转换为0-1概率值
        )
        
        

    def forward(self, x):
        # 提取多尺度特征
        features = self.feature_extractor(x)
        
        # 生成主高亮度特征图
        intensity_map = self.intensity_head(features)
        
        
        
        return intensity_map.clamp(0, 1)

class SemanticMaskExtractor(nn.Module):
    def __init__(self, 
                 target_classes: list = [1, 2,  9, 10, 11]):
        """
        精确语义类别掩膜提取器
        
        参数:
            target_classes: 需要提取的目标类别列表
        """
        super().__init__()
        self.target_classes = target_classes
        # 创建目标类别集用于快速查询
        self.target_set = set(target_classes)
        
    def forward(self, seg_map):
        """
        输入: 
            seg_map: 语义分割概率图 (B, N_class, H, W)
        输出: 
            mask: 目标类别的二值融合掩膜 (B, 1, H, W)
                规则: 如果最大概率通道属于target_classes，则掩膜为1，否则为0
        """
        
        B, N_class, H, W = seg_map.shape
        
        # 1. 获取每个像素点的预测类别
        pred_classes = seg_map.argmax(dim=1)  # (B, H, W)
        
        # 2. 创建目标类别掩膜
        target_mask = torch.zeros(B, H, W, dtype=torch.bool, device=seg_map.device)
        
        # 3. 检查每个目标类别
        for class_idx in self.target_classes:
            if 0 <= class_idx < N_class:
                # 标记属于该类别的像素
                target_mask |= (pred_classes == class_idx)
        
        # 4. 扩展维度并转为浮点类型
        return target_mask.unsqueeze(1).float()

class DR_mask(nn.Module):
    """
    语义引导融合图像重建网络
    输入:
        ir_img: 红外图像 (B, 3, H, W)
        seg_map: 语义分割图 (B, N_class, H, W)
        
    输出:
        dr_weight: 去退化区域权重 (B, 1, H, W)
    """
    def __init__(self):
        super().__init__()
        
        self.highlight_ir = HighIntensityCNN()
        self.dr_seg = SemanticMaskExtractor()

    def forward(self, ir_img,seg_map):
        ir_feature = self.highlight_ir(ir_img)
        dr_seg = self.dr_seg(seg_map)
        dr_weight = 1.0-ir_feature*dr_seg
        return dr_weight



class PointConv(nn.Module):
    """
    Point convolution block: input: x with size(B C H W); output size (B C1 H W)
    """
    def __init__(self, in_dim=64, out_dim=64,dilation=1, norm_layer=nn.BatchNorm2d):
        super(PointConv, self).__init__()
        self.kernel_size = 1
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dilation = dilation       
        conv_padding = (self.kernel_size // 2) * self.dilation        
                
        self.pconv = nn.Sequential(
                nn.Conv2d(self.in_dim, self.out_dim, self.kernel_size, padding=conv_padding, dilation = self.dilation),
                norm_layer(self.out_dim),
                nn.ReLU(inplace=True)
                )        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pconv(x) 
        return x


class MultiScaleFusion(nn.Module):
    def __init__(self,
                 in_channels=[64,128,320,512],
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=32,
                 align_corners=False):        
        super(MultiScaleFusion, self).__init__()
        self.align_corners = align_corners        
        self.in_channels = in_channels        
        F1_in_channels, F2_in_channels,F3_in_channels,F4_in_channels = self.in_channels
        embedding_dim = embed_dim
                
        self.PointConv_1_rgb = PointConv(in_dim=F1_in_channels, out_dim=embedding_dim)
        self.PointConv_1_ir = PointConv(in_dim=F1_in_channels, out_dim=embedding_dim)
        self.PointConv_2_rgb = PointConv(in_dim=F2_in_channels, out_dim=embedding_dim)
        self.PointConv_2_ir = PointConv(in_dim=F2_in_channels, out_dim=embedding_dim)
        self.PointConv_3_rgb = PointConv(in_dim=F3_in_channels, out_dim=embedding_dim)
        self.PointConv_3_ir = PointConv(in_dim=F3_in_channels, out_dim=embedding_dim)
        self.PointConv_4_rgb = PointConv(in_dim=F4_in_channels, out_dim=embedding_dim)
        self.PointConv_4_ir = PointConv(in_dim=F4_in_channels, out_dim=embedding_dim)
                
        self.CNN_vis = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4+3, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)                                                                              
                            )
        self.CNN_ir = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4+3, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)                                                               
                            )
    def forward(self, vis_img,ir_img, vis_feat, ir_feat):
        # len=4, 1/4,1/8,1/16,1/32
        F1_rgb, F2_rgb,F3_rgb,F4_rgb = vis_feat
        F1_ir, F2_ir,F3_ir,F4_ir = ir_feat
        

        F1_rgb_c = self.PointConv_1_rgb(F1_rgb)
        F1_rgb_c = F.interpolate(F1_rgb_c, size=vis_img.size()[2:],mode='bilinear',align_corners=self.align_corners)
                        
        F1_ir_c = self.PointConv_1_ir(F1_ir)
        F1_ir_c = F.interpolate(F1_ir_c, size=vis_img.size()[2:],mode='bilinear',align_corners=self.align_corners)
                      
        F2_rgb_c = self.PointConv_2_rgb(F2_rgb)
        F2_rgb_c = F.interpolate(F2_rgb_c, size=vis_img.size()[2:], mode='bilinear',align_corners=self.align_corners)
                      
        F2_ir_c = self.PointConv_2_ir(F2_ir)
        F2_ir_c = F.interpolate(F2_ir_c, size=vis_img.size()[2:],  mode='bilinear',align_corners=self.align_corners)
               
        F3_rgb_c = self.PointConv_3_rgb(F3_rgb)
        F3_rgb_c = F.interpolate(F3_rgb_c, size=vis_img.size()[2:],  mode='bilinear',align_corners=self.align_corners)
        
        F3_ir_c = self.PointConv_3_ir(F3_ir)
        F3_ir_c = F.interpolate(F3_ir_c, size=vis_img.size()[2:], mode='bilinear',align_corners=self.align_corners)
                
        F4_rgb_c = self.PointConv_4_rgb(F4_rgb)
        F4_rgb_c = F.interpolate(F4_rgb_c, size=vis_img.size()[2:],  mode='bilinear',align_corners=self.align_corners)

        F4_ir_c = self.PointConv_4_ir(F4_ir)
        F4_ir_c = F.interpolate(F4_ir_c, size=vis_img.size()[2:], mode='bilinear',align_corners=self.align_corners)

        F_1_4_cat_vis = torch.cat([F1_rgb_c, F2_rgb_c,F3_rgb_c,F4_rgb_c, vis_img], dim=1)
        F_1_4_cat_ir = torch.cat([F1_ir_c, F2_ir_c, F3_ir_c, F4_ir_c,ir_img], dim=1)
        vis_feature = self.CNN_vis(F_1_4_cat_vis)
        ir_feature = self.CNN_ir(F_1_4_cat_ir)
        return vis_feature,ir_feature

class Fusion_Head(nn.Module):
    def __init__(self,
                 embedding_dim=32,
                 norm_layer=nn.BatchNorm2d):        
        super(Fusion_Head, self).__init__()
        self.dr_weight =  DR_mask()
        self.feature =  MultiScaleFusion()  
        self.fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*2, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=embedding_dim, out_channels=1, kernel_size=1),
                            nn.Sigmoid()                                                                                
                            )
    def forward(self, vis_img,ir_img, seg,vis_feat, ir_feat):
        dr_weight = self.dr_weight(ir_img,seg)
        vis_feature,ir_feature = self.feature(vis_img,ir_img, vis_feat, ir_feat)
        ir_feature = ir_feature*dr_weight
        fuse = self.fuse(torch.cat([vis_feature,ir_feature], dim=1))
        return fuse
                         


