import torch
import torch.nn as nn
import torch.nn.functional as F
from .Seg_Head import Seg_Head
from .encoder_agg import RGBXTransformer
from .Fusion_Head import Fusion_Head
from torch.profiler import profile, record_function, ProfilerActivity




class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.backbone = RGBXTransformer(embed_dims=[64, 128, 320, 512])
        self.seg_head = Seg_Head(feature_strides=[4,8,16,32])
        self.fusion_head=Fusion_Head()
    def forward(self, rgb, ir):
        vis_features,ir_features,fuse_features= self.backbone(rgb,ir)
        
        seg = self.seg_head(fuse_features)
        
        fuse = self.fusion_head(rgb,ir,seg,vis_features,ir_features)
        
        return fuse,seg