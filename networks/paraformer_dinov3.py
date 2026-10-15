"""Paraformer with a foundation-model ViT (DINOv3 or Pixio) in place of the ImageNet-21k ViT-B/16.

Differences from the original (networks/vit_seg_modeling.VisionTransformer):
  * the Transformer branch sees the RAW RGB chip (ImageNet-normalised), not the CNN's 14x14 feature map,
    so the two branches are truly parallel;
  * the CNN branch (L2HNet) still takes all 4 bands and provides the full-res features + skip pyramid;
  * decoder / heads / PLAT loss are unchanged (DecoderCup, SegmentationHead reused as-is).
Backbones: any HF DINOv3 ViT id, e.g.
  facebook/dinov3-vitl16-pretrain-sat493m   (satellite, 0.6 m Maxar)   -> alias 'dinov3_sat'
  facebook/dinov3-vitl16-pretrain-lvd1689m  (standard web images)      -> alias 'dinov3_web'
  facebook/pixio-vitl16                     (MAE-style, 2B web images, CVPR 2026) -> alias 'pixio_l'
"""
import contextlib
import ml_collections
import torch, torch.nn as nn
from transformers import DINOv3ViTModel, DINOv3ViTConfig, PixioModel, PixioConfig
from .vit_seg_modeling import DecoderCup, SegmentationHead
from .vit_seg_modeling_L2HNet import L2HNet
import utils

BACKBONES = {
    "dinov3_sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "dinov3_web": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "pixio_l": "facebook/pixio-vitl16",
}
# Normalisation per pretraining corpus (facebookresearch/dinov3 README):
#   LVD-1689M web models  -> ImageNet mean/std
#   SAT-493M satellite    -> mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)
NORM = {
    "web": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "sat": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
}
def norm_for(name):
    # Pixio uses ImageNet mean/std (facebookresearch/pixio README), like the web DINOv3
    return NORM["sat"] if "sat493m" in name or name == "dinov3_sat" else NORM["web"]

def is_pixio(name):
    return "pixio" in name

def pixio_l16_config():
    """Pixio ViT-L/16 shape, for shape tests without the gated weights."""
    return PixioConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, mlp_ratio=4,
                       patch_size=16, image_size=256, n_cls_tokens=8)

def vitl16_config():
    """DINOv3 ViT-L/16 shape, for shape tests without the gated weights."""
    return DINOv3ViTConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                           intermediate_size=4096, patch_size=16, num_register_tokens=4, image_size=224)

class ParaformerDINOv3(nn.Module):
    def __init__(self, backbone="dinov3_sat", cnn_width=128, num_classes=17, freeze_backbone=True, random_init=False):
        super().__init__()
        name = BACKBONES.get(backbone, backbone)
        self.cnn = L2HNet(width=cnn_width)
        if is_pixio(name):
            self.vit = PixioModel(pixio_l16_config()) if random_init else PixioModel.from_pretrained(name)
        else:
            self.vit = DINOv3ViTModel(vitl16_config()) if random_init else DINOv3ViTModel.from_pretrained(name)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.vit.parameters(): p.requires_grad = False
        D = self.vit.config.hidden_size
        c = self.vit.config                                              # prefix tokens to drop before the patch grid:
        self.n_prefix = c.n_cls_tokens if is_pixio(name) else 1 + c.num_register_tokens   # Pixio: 8 class tokens; DINOv3: CLS + registers
        self.decoder = DecoderCup(ml_collections.ConfigDict({"hidden_size": D}))
        self.segmentation_head1 = SegmentationHead(in_channels=cnn_width + 1, out_channels=num_classes, kernel_size=3)
        self.segmentation_head2 = SegmentationHead(in_channels=256, out_channels=num_classes, kernel_size=3)
        # de-normalise Chesapeake-standardised input back to 0-255, then ImageNet-normalise RGB for DINOv3
        self.register_buffer("ches_mean", torch.tensor(utils.IMAGE_MEANS[:3], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("ches_std", torch.tensor(utils.IMAGE_STDS[:3], dtype=torch.float32).view(1, 3, 1, 1))
        vit_mean, vit_std = norm_for(name)
        self.register_buffer("im_mean", torch.tensor(vit_mean).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor(vit_std).view(1, 3, 1, 1))
        self.vit_norm = (vit_mean, vit_std)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone: self.vit.eval()
        return self

    def rgb_for_vit(self, x):
        rgb = x[:, :3] * self.ches_std + self.ches_mean            # NAIP bands are R,G,B,NIR
        return (rgb / 255.0 - self.im_mean) / self.im_std

    def forward(self, x):
        _, features = self.cnn(x)                                   # [full-res, 14x14, 56x56, 112x112]
        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            tokens = self.vit(pixel_values=self.rgb_for_vit(x)).last_hidden_state
        patch_tokens = tokens[:, self.n_prefix:]                    # (B, 196, D) for a 224 chip
        x1, x2 = self.decoder(patch_tokens, features)
        return self.segmentation_head1(x1), self.segmentation_head2(x2)

    def load_from(self, weights):  # API parity with VisionTransformer; DINOv3 weights come from HF instead
        pass
