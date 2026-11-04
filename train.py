import argparse
import numpy as np
from torchvision import transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
import math
import random
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile


from net_.modules import ToTensorKeepValues
from net_.NetWork import NetWork
import utils
import losses

ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_seed(seed, rank):
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dataset', type=str, default='./dataset/train_FMB/Visible/')
    parser.add_argument('--ir_dataset', type=str, default='./dataset/train_FMB/Infrared/')
    parser.add_argument('--label_dataset', type=str, default='./dataset/train_FMB/Label/')
    
    parser.add_argument('--pretrained_path', type=str, default='./pretrained/mit_b4.pth')
    parser.add_argument('--log_dir', type=str, default='./logs/log')
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--patience', type=int, default=30)
    
  
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

  
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
    else:
        print("未检测到分布式环境，启动单卡训练模式...")
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False

    set_seed(args.seed, rank)

    trans = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,),(0.5,))
    ])
    trans_seg = transforms.Compose([ToTensorKeepValues()])
    
    full_dataset = utils.Customdataset(
        transform=trans, transform_seg=trans_seg, 
        rgb_dataset=args.rgb_dataset, 
        ir_dataset=args.ir_dataset, 
        label_dataset=args.label_dataset, 
        Resize_H=args.resize_h, Resize_W=args.resize_w
    )

    val_split_ratio = 0.1
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split_ratio * dataset_size))

    random.seed(args.seed) 
    random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                  shuffle=shuffle, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                                shuffle=False, num_workers=4, pin_memory=True)

    if rank == 0:
        print(f"--- 模式: {'分布式 DDP' if is_distributed else '单卡'} ---")
        print(f"训练集规模: {len(train_dataset)} | 验证集规模: {len(val_dataset)}")

    model = NetWork().to(device)
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model.backbone.init_weights(pretrained=args.pretrained_path)

    fusion_head_ids = list(map(id, model.fusion_head.parameters()))
    seg_head_ids = list(map(id, model.seg_head.parameters()))

    base_params = [
        p for p in model.parameters() 
        if id(p) not in fusion_head_ids and id(p) not in seg_head_ids
    ]

    param_groups = [
        {
            'params': base_params, 
            'lr': args.lr* 0.2, 
            'name': 'backbone'
        },
        {
            'params': model.seg_head.parameters(), 
            'lr': args.lr , 
            'name': 'seg_head'
        },
        {
            'params': model.fusion_head.parameters(), 
            'lr': args.lr, 
            'name': 'fusion_head'
        }
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=0.01)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    min_lr_ratio = 1e-6 / args.lr
    def lr_lambda(current_epoch):
        if current_epoch < args.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, args.warmup_epochs))
        else:
            progress = float(current_epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    imgloss = losses.Loss().to(device)
    writer = SummaryWriter(log_dir=args.log_dir) if rank == 0 else None
    best_val_loss = float('inf') 
    early_stop_counts = 0
    flag = 0 

    if rank == 0: os.makedirs("./saved_models", exist_ok=True)

    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        loss_keys = ['total', 'con', 'grad', 'real_ce', 'aux_ce']
        train_losses_sum = {k: torch.tensor(0.0).to(device) for k in loss_keys}

        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", disable=(rank != 0))

        for rgb_target, ir_target, label in train_dataloader:
            optimizer.zero_grad()
            r_img, i_img, l_img = rgb_target.to(device), ir_target.to(device), label.to(device)
            
            fuse, seg = model(r_img, i_img)
            con_loss, grad_loss, real_ce, aux_ce, _ = imgloss(r_img, i_img, fuse, seg, l_img, flag) 
            
            total_loss = (con_loss + 0.75*grad_loss) + 0.25*(real_ce + aux_ce)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            curr_batch_size = r_img.size(0)
            train_losses_sum['total'] += total_loss.detach() * curr_batch_size
            train_losses_sum['con'] += con_loss.detach() * curr_batch_size
            train_losses_sum['grad'] += grad_loss.detach() * curr_batch_size
            train_losses_sum['real_ce'] += real_ce.detach() * curr_batch_size
            train_losses_sum['aux_ce'] += aux_ce.detach() * curr_batch_size

            if rank == 0:
                pbar.set_postfix({'Total': f"{total_loss.item():.4f}", 'LR': f"{optimizer.param_groups[0]['lr']:.6f}"})
                pbar.update(1)
        
        if rank == 0: pbar.close()

        model.eval()
        val_losses_sum = {k: torch.tensor(0.0).to(device) for k in loss_keys}
        
        with torch.no_grad():
            for rgb, ir, lb in val_dataloader:
                rv_img, iv_img, lv_img = rgb.to(device), ir.to(device), lb.to(device)
                fv, sv = model(rv_img, iv_img)
                v_con, v_grad, v_real_ce, v_aux_ce, _ = imgloss(rv_img, iv_img, fv, sv, lv_img, flag)
                v_total = (v_con +  0.75*v_grad) + 0.25*(v_real_ce + v_aux_ce)
                
                v_batch_size = rv_img.size(0)
                val_losses_sum['total'] += v_total.detach() * v_batch_size
                val_losses_sum['con'] += v_con.detach() * v_batch_size
                val_losses_sum['grad'] += v_grad.detach() * v_batch_size
                val_losses_sum['real_ce'] += v_real_ce.detach() * v_batch_size
                val_losses_sum['aux_ce'] += v_aux_ce.detach() * v_batch_size

        final_train_losses = {}
        final_val_losses = {}

        for k in loss_keys:
            if is_distributed:
                dist.all_reduce(train_losses_sum[k], op=dist.ReduceOp.SUM)
                dist.all_reduce(val_losses_sum[k], op=dist.ReduceOp.SUM)
            
            final_train_losses[k] = train_losses_sum[k].item() / len(train_dataset)
            final_val_losses[k] = val_losses_sum[k].item() / len(val_dataset)

        if rank == 0:
            for k in loss_keys:
                writer.add_scalars(f'Loss/{k}', {'train': final_train_losses[k], 'val': final_val_losses[k]}, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            current_val_total = final_val_losses['total']
            if current_val_total < best_val_loss:
                best_val_loss = current_val_total
                early_stop_counts = 0
                save_obj = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(save_obj, "./saved_models/best_model.pth")
                print(f"--- 记录更新 | Epoch {epoch+1} | Val Loss: {current_val_total:.4f} ---")
            else:
                early_stop_counts += 1
                print(f"--- 未改善 | EarlyStop: {early_stop_counts}/{args.patience} ---")
        
        scheduler.step()
        
        if is_distributed:
            stop_signal = torch.tensor(1.0 if early_stop_counts >= args.patience else 0.0).to(device)
            dist.all_reduce(stop_signal, op=dist.ReduceOp.SUM)
            if stop_signal > 0: break
        else:
            if early_stop_counts >= args.patience: break

    if rank == 0: writer.close()
    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
    # python -m torch.distributed.run --nproc_per_node=2 train.py --batch_size 3