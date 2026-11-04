import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from sklearn.decomposition import PCA
from net_.NetWork import NetWork
from torchvision.utils import save_image, make_grid 
import utils
import torch.nn.functional as F
import argparse
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np

def apply_colormap_to_all_channels(tensor_feature, colormap=cv2.COLORMAP_JET):
    """对每个通道应用伪彩色映射，并生成网格图"""
    batch, channels, h, w = tensor_feature.shape
    colored_list = []
    
    for c in range(channels):
        channel_data = tensor_feature[0, c].cpu().numpy()
        
        normalized = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
        normalized_uint8 = (normalized * 255).astype(np.uint8)
        
        colored = cv2.applyColorMap(normalized_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        colored_tensor = torch.from_numpy(colored_rgb).permute(2, 0, 1).float() / 255.0
        colored_list.append(colored_tensor)
    
    grid_input = torch.stack(colored_list)
    grid = make_grid(grid_input, nrow=8, padding=10, normalize=False)
    return grid
def run():
    parser = argparse.ArgumentParser()
    #results_path setting
    parser.add_argument('--out_path', type=str, default='./fusion_outputs/', help='path of fused image')
    parser.add_argument('--rgb_path', type=str, default='./results_FMB/', help='path of fused RGB image')
    parser.add_argument('--seg_output_path', type=str, default='./seg_outputs/', help='path of seg image')

    #dataset setting
    parser.add_argument('--ir_images', type=str, default='./dataset/test_FMB/Infrared/', help='path of infrared images')
    parser.add_argument('--vi_images', type=str, default='./dataset/test_FMB/Visible/', help='path of visible images')
    
    #model_path setting
    parser.add_argument('--model_name', type=str, default='best_model', help='best weights')
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    # device setting for gpu users
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    fusion_model = torch.nn.DataParallel(NetWork(), device_ids=[0])

    state_dict = torch.load("./saved_models/%s.pth" % (args.model_name))
    fusion_model.module.load_state_dict(state_dict)

    fusion_model.cuda().eval()

    with torch.no_grad():
        ir_files = [f for f in os.listdir(args.ir_images) if f.endswith('.png') or f.endswith('.jpg')]

        for index, ir_filename in tqdm(enumerate(ir_files), total=len(ir_files), desc="Processing Fused Images_Y to /fusion_outputs/"):
            vi_filename = ir_filename
            ir_path = os.path.join(args.ir_images, ir_filename)
            vi_path = os.path.join(args.vi_images, vi_filename)

            if os.path.isfile(ir_path) and os.path.isfile(vi_path):
                real_ir_imgs = utils.get_test_images(ir_path, height=600, width=800)
                real_rgb_imgs = utils.get_test_images(vi_path, height=600, width=800)
            
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                fuse,seg = fusion_model(real_rgb_imgs.cuda(), real_ir_imgs.cuda())
                fuse = F.interpolate(fuse, size=(600, 800), mode='bilinear', align_corners=False)
                seg= F.interpolate(seg, size=(600, 800), mode='bilinear', align_corners=False)
                
                
                seg = utils.SegmentationVisualizer(seg)
                seg = seg.visualize()
                seg = seg.squeeze(0)
                seg = seg.permute(1,2,0)
                seg = seg.cpu().numpy().astype('uint8')

                save_image(fuse, os.path.join(args.out_path, f"{ir_filename}"), normalize=True)
                Image.fromarray(seg).save(os.path.join(args.seg_output_path, f"{ir_filename}"))

    fusion_outputs_dir = args.out_path
    output_dir = args.rgb_path

    os.makedirs(output_dir, exist_ok=True)

    vi_files = [filename for filename in os.listdir(args.vi_images) if
                filename.endswith('.png') or filename.endswith('.jpg')]

    total_files = len(vi_files)

    for index, filename in tqdm(enumerate(vi_files), total=total_files, desc="Processing Fused Images_RGB to /result_FMB/"):
        vi_image = cv2.imread(os.path.join(args.vi_images, filename))

        ycrcb_image = cv2.cvtColor(vi_image, cv2.COLOR_RGB2YCrCb)

        gray_image_path = os.path.join(fusion_outputs_dir, f"{filename}")

        if os.path.exists(gray_image_path):
            gray_image = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)

            if gray_image.shape[:2] == ycrcb_image[:, :, 0].shape:
                ycrcb_image[:, :, 0] = gray_image

                new_rgb_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)

                output_filename = f'{filename}'
                cv2.imwrite(os.path.join(output_dir, output_filename), new_rgb_image)

    print('Done......')


if __name__ == '__main__':
    run()
  