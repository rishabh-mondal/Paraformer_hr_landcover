"""Evaluate a Paraformer checkpoint against CCLC ground truth with the 4-class unification.

usage: python evaluate.py --model_path <ckpt.pth> --list_dir <csv with image_fn,gt_fn> --gpu 1 --out_dir <dir> [--save_pred]

Inference follows test.py exactly (224 chips, stride 112, centre-weighted blending, branch outputs averaged).
Predictions (17 NLCD indices) and CCLC (6 classes) are both mapped to 4 unified classes with the paper's Table S1
(see eval_metrics.py; --mapping legacy reproduces our earlier barren-ignored numbers).
A full-resolution 17 x 16 count table per tile is saved to raw_counts.npz, so rescore.py can re-map without inference.
mIoU is reported per state (confusion matrix pooled over that state's tiles) and as the mean over states, like Table 1.
"""
import argparse, os, json, time, sys
import numpy as np, pandas as pd, rasterio, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from networks.vit_seg_modeling import VisionTransformer as ViT_seg, CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling_L2HNet import L2HNet
import utils

CHIP_SIZE, PADDING = 224, 112
HALF_PADDING = PADDING // 2
CHIP_STRIDE = CHIP_SIZE - PADDING
import eval_metrics as em

class TileDataset(Dataset):
    def __init__(self, fn):
        with rasterio.open(fn) as f:
            self.data = np.rollaxis(f.read(), 0, 3); h, w = f.height, f.width
        self.coords = [(y, x) for y in list(range(0, h - CHIP_SIZE, CHIP_STRIDE)) + [h - CHIP_SIZE]
                              for x in list(range(0, w - CHIP_SIZE, CHIP_STRIDE)) + [w - CHIP_SIZE]]
    def __len__(self): return len(self.coords)
    def __getitem__(self, i):
        y, x = self.coords[i]
        img = (self.data[y:y+CHIP_SIZE, x:x+CHIP_SIZE] - utils.IMAGE_MEANS) / utils.IMAGE_STDS
        return torch.from_numpy(np.rollaxis(img, 2, 0).astype(np.float32)), np.array((y, x))

@torch.no_grad()
def predict_tile(model, image_fn, num_classes, batch_size=32):
    with rasterio.open(image_fn) as f: H, W, profile = f.height, f.width, f.profile.copy()
    ds = TileDataset(image_fn)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    out = torch.zeros((num_classes, H, W), dtype=torch.float32, device="cuda")
    cnt = torch.zeros((H, W), dtype=torch.float32, device="cuda")
    kernel = torch.ones((CHIP_SIZE, CHIP_SIZE), device="cuda"); kernel[HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING] = 5
    for data, coords in dl:
        o1, o2 = model(data.cuda(non_blocking=True))
        prob = F.softmax((o1 + o2) / 2, dim=1)
        for j in range(prob.shape[0]):
            y, x = int(coords[j][0]), int(coords[j][1])
            out[:, y:y+CHIP_SIZE, x:x+CHIP_SIZE] += prob[j] * kernel
            cnt[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += kernel
    return (out / cnt).argmax(0).to(torch.uint8).cpu().numpy(), profile

def state_of(fn): return os.path.basename(os.path.dirname(fn))[:2]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True); p.add_argument("--list_dir", required=True)
    p.add_argument("--out_dir", required=True); p.add_argument("--gpu", default="0")
    p.add_argument("--CNN_width", type=int, default=128); p.add_argument("--num_classes", type=int, default=17)
    p.add_argument("--backbone", default="vit_b16")
    p.add_argument("--mapping", default="paper", choices=sorted(em.MAPPINGS), help="class unification (default: paper Table S1)")
    p.add_argument("--save_pred", action="store_true", help="write 17-index prediction GeoTIFFs (large)")
    p.add_argument("--save_png", type=int, default=0, metavar="F",
                   help="write the 4-class unified prediction as a small palette PNG, downsampled by factor F (e.g. 4); 0 = off")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out_dir, exist_ok=True)

    if args.backbone == "vit_b16":
        cfg = CONFIGS_ViT_seg["ViT-B_16"]; cfg.n_classes = args.num_classes; cfg.patches.grid = (224 // 16, 224 // 16)
        model = ViT_seg(cfg, backbone=L2HNet(width=args.CNN_width), img_size=224, num_classes=args.num_classes).cuda()
    else:
        from networks.paraformer_dinov3 import ParaformerDINOv3
        model = ParaformerDINOv3(backbone=args.backbone, cnn_width=args.CNN_width, num_classes=args.num_classes, freeze_backbone=True).cuda()
    model.load_state_dict(torch.load(args.model_path, map_location="cuda")); model.eval()

    df = pd.read_csv(args.list_dir)
    P_LUT, G_LUT = em.luts(args.mapping)
    states, tiles, raws = [], [], []
    t0 = time.time()
    for i, row in df.iterrows():
        st = state_of(row.image_fn); tile = os.path.basename(row.image_fn)
        pred, profile = predict_tile(model, row.image_fn, args.num_classes)
        with rasterio.open(row.gt_fn) as f: gt = f.read(1)
        raw = em.raw_counts(pred, gt); states.append(st); tiles.append(tile); raws.append(raw)
        s = em.scores(em.unify(raw, args.mapping))
        print(f"[{i+1}/{len(df)}] {st} {tile}  mIoU={s['mIoU']:.2f}  OA={s['OA']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        if args.save_png:
            from PIL import Image
            os.makedirs(os.path.join(args.out_dir, "png"), exist_ok=True)
            pred_u = P_LUT[pred[::args.save_png, ::args.save_png]]                       # nearest-neighbour decimation
            im = Image.fromarray(np.ascontiguousarray(pred_u), mode="P")
            pal = [156, 156, 156, 38, 115, 0, 163, 255, 115, 0, 197, 255] + [255, 255, 255] * 252   # built-up, tree, low veg, water; rest white
            im.putpalette(pal)
            im.save(os.path.join(args.out_dir, "png", tile.replace("_naip-new.tif", "_pred4.png")), optimize=True)
        if args.save_pred:
            profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw")
            with rasterio.open(os.path.join(args.out_dir, tile.replace("naip", "pred")), "w", **profile) as f:
                f.write(pred, 1); f.write_colormap(1, utils.LABEL_IDX_COLORMAP)
        np.savez_compressed(os.path.join(args.out_dir, "raw_counts.npz"), states=np.array(states), tiles=np.array(tiles), raws=np.stack(raws))

    result, per_tile = em.summarize(states, tiles, raws, args.mapping)
    result = {"model_path": args.model_path, "list_dir": args.list_dir, **result}
    json.dump(result, open(os.path.join(args.out_dir, "results.json"), "w"), indent=2)
    pd.DataFrame(per_tile).to_csv(os.path.join(args.out_dir, "per_tile.csv"), index=False)
    print("\n=== mIoU (%) per state ===")
    for st, v in result["per_state"].items(): print(f"{st}: {v['mIoU']:.2f}   " + "  ".join(f"{c}={x:.1f}" for c, x in v["IoU"].items()))
    print(f"Average over states: {result['mean_over_states_mIoU']:.2f}   pooled: {result['pooled']['mIoU']:.2f}   OA pooled: {result['pooled']['OA']:.2f}")
