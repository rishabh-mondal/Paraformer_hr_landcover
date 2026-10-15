import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_dataset
import os
from networks.vit_seg_modeling_L2HNet import L2HNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Chesapeake', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--savepath', type=str)
parser.add_argument('--gpu', type=str, help='Select GPU number to train' )
parser.add_argument('--list_dir', type=str, default='./dataset/CSV_list/Chesapeake_all_train.csv', help='CSV with image_fn,label_fn columns')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help="'adamw' = paper setting (AdamW, lr drops 10%% after 8 stalled epochs); 'sgd' = original repo code")
parser.add_argument('--backbone', type=str, default='vit_b16', help="'vit_b16' = original ImageNet-21k ViT-B/16; 'dinov3_sat' / 'dinov3_web' or any HF DINOv3 id")
parser.add_argument('--freeze_backbone', action='store_true', help='freeze the DINOv3 ViT (CNN branch, decoder and heads still train)')
parser.add_argument('--num_chips_per_tile', type=int, default=50, help='random 224x224 chips sampled per tile per epoch')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if __name__ == "__main__":
    vit_patches_size=16
    img_size=224
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Chesapeake': {  # default dataset as a example
            'list_dir': args.list_dir,
            'num_classes': 17
        }
    }# Create a config to your own dataset here
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = args.savepath 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if args.backbone == 'vit_b16':
        config_vit = CONFIGS_ViT_seg["ViT-B_16"]
        config_vit.n_classes = args.num_classes
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        net = ViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width),img_size=img_size, num_classes=config_vit.n_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        from networks.paraformer_dinov3 import ParaformerDINOv3
        net = ParaformerDINOv3(backbone=args.backbone, cnn_width=args.CNN_width, num_classes=args.num_classes,
                               freeze_backbone=args.freeze_backbone).cuda()
    print("trainable params: %.1fM / %.1fM" % (sum(p.numel() for p in net.parameters() if p.requires_grad)/1e6, sum(p.numel() for p in net.parameters())/1e6))
    trainer_dataset(args, net, snapshot_path)
