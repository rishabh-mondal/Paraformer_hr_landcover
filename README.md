# Paraformer: Updating Large-scale High-resolution Geographical Maps from Limited Historical Labels

## File guide

This branch adds DINOv3 and Pixio backbones, checkpoint evaluation, result collection, and comparison figures to the original Paraformer code.

### Training and model files

| File | What it does |
| --- | --- |
| [train.py](train.py) | Starts training from a dataset CSV. Selects the original ViT-B/16 or a DINOv3/Pixio backbone, optimizer, backbone freezing, and chips sampled per tile. |
| [trainer.py](trainer.py) | Streams image tiles, samples training chips, computes the two branch losses, and runs optimization. Handles invalid labels and non-finite losses, logs training progress, and saves checkpoints. |
| [utils.py](utils.py) | Defines image normalization, NLCD class indices, and label colors. Maps unknown uint8 label values to the ignored class. |
| [networks/paraformer_dinov3.py](networks/paraformer_dinov3.py) | Implements Paraformer with DINOv3 satellite/web or Pixio backbones alongside the four-band CNN branch. Handles RGB normalization and optional backbone freezing. |
| [networks/vit_seg_modeling.py](networks/vit_seg_modeling.py) | Implements the original ViT branch, shared decoder, segmentation heads, and pretrained ViT weight loading. |
| [networks/vit_seg_modeling_L2HNet.py](networks/vit_seg_modeling_L2HNet.py) | Implements the L2HNet CNN branch and its multiscale features. |
| [networks/vit_seg_configs.py](networks/vit_seg_configs.py) | Defines ViT-B/16 and testing configurations. |

### Evaluation, results, and figures

| File | What it does |
| --- | --- |
| [test.py](test.py) | Runs the original ViT-B/16 sliding-window inference workflow and writes prediction rasters. |
| [evaluate.py](evaluate.py) | Evaluates a checkpoint against CCLC ground truth using four unified classes. Saves raw counts, per-tile CSV scores, and JSON results; optionally saves prediction GeoTIFFs or downsampled PNGs. |
| [eval_metrics.py](eval_metrics.py) | Provides the `paper` and `legacy` class mappings, confusion matrices, class IoU, mIoU, and overall accuracy. Computes per-state, mean-over-states, and pooled results. |
| [rescore.py](rescore.py) | Recomputes metrics from saved `raw_counts.npz` without model inference. Prints results by default; `--write` updates the JSON and per-tile CSV files. |
| [append_history.py](append_history.py) | Adds checkpoint results to a history CSV. Replaces an existing row for the same epoch and class mapping. |
| [collect_results.py](collect_results.py) | Combines test histories using the `paper` mapping for the baseline and two DINOv3 runs into `runs/test_results_tableS1.csv`. |
| [make_test_figures.py](make_test_figures.py) | Creates side-by-side RGB, ground-truth, and three-model prediction figures for a chosen epoch. Writes per-tile images and an overview with one representative tile per state. Requires predictions saved with `evaluate.py --save_png 4`. |

### Shell workflows

| File | What it does |
| --- | --- |
| [eval_latest.sh](eval_latest.sh) | Copies `latest.pth` to a temporary snapshot, evaluates validation and test sets on GPUs 1 and 2, updates their histories, and removes the snapshot. Accepts a run directory and backbone. |
| [eval_epoch_all.sh](eval_epoch_all.sh) | Evaluates one supplied epoch for the baseline and two DINOv3 runs on GPUs 0–2. Saves downsampled prediction PNGs and updates test histories. |
| [eval_all_epochs_gpu0.sh](eval_all_epochs_gpu0.sh) | Queues epochs 0, 10, ..., 90, and 99 for all three runs sequentially on GPU 0. Waits for checkpoints, skips recorded `paper` results, updates the combined CSV, and invokes slide generation. |
| [rescore_paper_all.sh](rescore_paper_all.sh) | Runs a predefined set of checkpoint evaluations with the `paper` mapping across GPUs 0–3 and updates histories. Re-runs inference through `evaluate.py`; epoch-50 jobs also save PNGs. |
| [after_rescore.sh](after_rescore.sh) | Waits for the rescore log to report completion, then rebuilds epoch-50 comparison figures and invokes slide generation. |
| [pixio_smoke.sh](pixio_smoke.sh) | Waits for access to the gated Pixio model, checks weight loading, trains for one epoch on a small dataset, and evaluates one test tile. Removes smoke-test checkpoints and keeps logs. |

Typical workflow: `train.py` → `evaluate.py` → `append_history.py` → `collect_results.py`. Use `make_test_figures.py` after saving prediction PNGs, or `rescore.py` to change the evaluation mapping without repeating inference.

The shell workflows contain machine-specific project paths, Python environment paths, run names, and GPU assignments; edit these for your setup. The slide-generation steps call `../make_slides.py`, which is outside this repository. Dataset CSVs and generated `runs/` outputs are kept locally and are not included in this branch's changes.

## Project overview

Large-scale high-resolution (HR) mapping is a vital task to survey the Earth's surface and resolve many challenges facing humanity. 
However, it is still a non-trivial task hindered by complex ground details, various landforms, and the scarcity of accurate training labels over a wide-span geographic area. 
To address these limitations, we propose an efficient, weakly supervised framework (Paraformer), a.k.a Low-to-High Network (L2HNet) v2, to guide large-scale HR mapping with easy-access annotation data, e.g., historical map at a low resolution or Volunteer Geographic Information data. 🌟:**Currently, the framework supports mappings of land cover, land use, and building function**.

The Paraformer is accepted by **IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** 2024, ranking as a **:rocket:Highlight:rocket:** paper (Top 2.6%) with a score of 5/5/4!

Contact me at ashelee@whu.edu.cn
* [**Paper**](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Learning_without_Exact_Guidance_Updating_Large-scale_High-resolution_Land_Cover_Maps_CVPR_2024_paper.pdf)
* [**My homepage**](https://lizhuohong.github.io/lzh/)
  
Our related works:
* [**SinoBF-1**](https://www.nature.com/articles/s41467-026-69589-5): accepted by Nature Communications 2026, individual building function map of China.[**Code**](https://github.com/LiZhuoHong/SinoBF-1/)
* [**SinoLC-1**](https://essd.copernicus.org/articles/15/4749/2023/): accepted by ESSD in 2023, the first 1-m resolution national-scale land-cover map of China.[**Data**](https://zenodo.org/record/7821068)
* [**BuildingMap**](https://ieeexplore.ieee.org/document/10641437): accepted by IGARSS 2024 (Oral), To identify every building's function in urban area.[**Data**](https://github.com/LiZhuoHong/BuildingMap/)
* [**SegLand**](https://openaccess.thecvf.com/content/CVPR2024W/L3D-IVU/papers/Li_Generalized_Few-Shot_Meets_Remote_Sensing_Discovering_Novel_Classes_in_Land_CVPRW_2024_paper.pdf): accepted by CVPRW 2024 (Oral) and won 1st place in OpenEarthMap Challenge, discovering novel classes in land-cover mapping.[**Code**](https://github.com/LiZhuoHong/SegLand)
* [**L2HNet V1**](https://www.sciencedirect.com/science/article/abs/pii/S0924271622002180): accepted by ISPRS P&RS in 2022. The low-to-high network for HR mapping using LR labels.

## News! New work accepted by Nature Communications 2026!
-------
**Check our paper: 
[**SinoBF-1**](https://www.nature.com/articles/s41467-026-69589-5)**

**Check the GitHub page:
[**Code**](https://github.com/LiZhuoHong/SinoBF-1/)**

**Check the data (Building function map for 109 cities in China):
[**Data**](https://zenodo.org/records/17844789)**

<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/Building_function-mapping-l.png" width="70%">
In our latest work, we utilize the framework to present **the nationwide building-level functional map of urban China**, processing over 69 TB of satellite data, including *1-meter Google Earth optical imagery*, *10-meter nighttime lights (SGDSAT-1)*, and *building height data (CNBH-10m)*. All data will be open access soon!

The mapping process contains segmentation and object classification parts that are shown below:

<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/Building_mapping_result-l2.png" width="70%">

------- 


## Training Instructions

* **To train and test the Paraformer on the default Chesapeake Bay dataset, follow these steps:**
1. Download the imagenet21k ViT pre-train model at [**Pre-train ViT**](https://drive.google.com/file/d/10Ao75MEBlZYADkrXE4YLg6VObvR0b2Dr/view?usp=sharing) and put it at *"./networks/pre-train_model/imagenet21k"*
   
2. If you want to run the code with the default Chesapeake dataset, we provide example data for the state of New York. Download the dataset at [**Baidu cloud**](https://pan.baidu.com/s/1FOCRmWenNmJ9mYDKBK_uXQ?pwd=2024) and put them at *"./dataset/Chesapeake_NewYork_dataset"*.
   
3. Run the "Train" command:
   ```bash
   python train.py --dataset Chesapeake --batch_size 10 --max_epochs 100 --savepath *save path of your folder* --gpu 0
4. After training, run the "Test" command:
   ```bash
   python test.py --dataset Chesapeake --model_path *The path of trained .pth file* --save_path *To save the inferred results* --gpu 0
   
* **To train and test the framework on your dataset:**

1. Generate a train and test list (.csv) of your dataset (an example is in the "dataset" folder).
2. Change the label class and colormap in the "utils.py" file.
3. Add your dataset_config in the "train.py" and "test.py" files.
4. Run the command above by changing the dataset name.
   
The Chesapeake Dataset
-------
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/The%20Chesapeake%20Dataset.png" width="70%">
The Chesapeake Bay dataset, grouped by Microsoft Research, contains 1-meter resolution images and a 30-meter resolution land-cover product as the training data pairs and also contains a 1-meter resolution ground reference for assessment. The figure illustrates the location, Digital Elevation Model (DEM), numbers of the tiles, and data samples of the Chesapeake Bay dataset. 

* **The HR remote sensing images** with 1-meter resolution were captured by the airborne platform of the U.S. Department of Agriculture’s National Agriculture Imagery Program (NAIP). The images contained four bands of red, green, blue, and near-infrared.

* **The rough historical land-cover products** with 30-meter resolution were collected from the National Land Cover Database of the United States Geological Survey (USGS). The NLCD data contains 16 land-cover types and is utilized as the labels during the training process of the proposed Paraformer framework.

* **The HR ground references** with 1-meter resolution were obtained from the Chesapeake Bay Conservancy Land Cover (CCLC) project. The CCLC data were interpreted based on the 1-meter NAIP imagery and LiDAR data containing six land-cover types. In this paper, the CCLC data were only used as the ground reference for quantitative and qualitative assessment and were not involved in the framework training or optimization process. 

The data can be downloaded at Microsoft's website: [**Chesapeake dataset**](https://lila.science/datasets/chesapeakelandcover)

The Poland Dataset
-------
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/The%20Poland%20dataset.png" width="70%">
The Poland dataset contained 14 Provinces of Poland, including the Provinces of Pomorskie, Lódzkie, Lubuskie, Dolnoslaskie, etc. The figure demonstrates the location, DEM, numbers of the tiles, and data samples of the Poland dataset. 

* **The HR remote sensing images** with 0.25-meter and 0.5-meter resolution were collected from the LandCover.ai dataset where the image sources are from the public geodetic resource used in the Land Parcel Identification System (LPIS). The images contained three bands of red, green, and blue.

* **The rough historical labeled data** with 10-meter resolution were collected from three types of global land-cover products which were (1) The FROM_GLC10 provided by the Tsinghua University, (2) The ESA_WorldCover v100 provided by the European Space Agency (ESA), and (3) The ESRI 10-meter global land cover (abbreviated as ESRI_GLC10) provided by the ESRI Inc. and IO Inc. The 30-meter resolution labeled data were collected from the 30-meter global land-cover product GLC_FCS30 provided by the Chinese Academy of Sciences (CAS).

* **The HR ground references** were obtained from the OpenEarthMap dataset provided by the University of Tokyo. The ground references were interpreted based on the 0.25-meter and 0.5-meter resolution LPIS imagery and contained five land-cover types. 

The data can be downloaded at [**Poland dataset**](https://drive.google.com/file/d/1qz1r5IQ-bkpUJN52GeGwvCFcbsaRu1Gs/view?usp=sharing)

The SinoLC-1 Dataset
-------
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/Fig/The%20SinoLC-1%20dataset.png" width="70%">
Based on our previous work on SinoLC-1 (i.e., the first 1-m land-cover map of China), we regard the intersected results of three 10-m land-cover products (ESA_GLC10, Esri_GLC10, and FROM_GLC10) as the LR training labels of 1-m Google Earth images. The Paraformer refines a more accurate urban pattern. For the whole of Wuhan City, the reported overall accuracy (OA) of SinoLC-1 is 72.40%. The updated results of the proposed Paraformer reach 74.98% with a 2.58% improvement.

The data can be downloaded at [**SinoLC-1 dataset**](https://doi.org/10.5281/zenodo.7707461)

Citation
-------
   ```bash
@article{li2022breaking,
  title={Breaking the resolution barrier: A low-to-high network for large-scale high-resolution land-cover mapping using low-resolution labels},
  author={Li, Zhuohong and Zhang, Hongyan and Lu, Fangxiao and Xue, Ruoyao and Yang, Guangyi and Zhang, Liangpei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={192},
  pages={244--267},
  year={2022},
  publisher={Elsevier}
}

@InProceedings{Li_2024_CVPR,
    author    = {Li, Zhuohong and He, Wei and Li, Jiepan and Lu, Fangxiao and Zhang, Hongyan},
    title     = {Learning without Exact Guidance: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27717-27727}
}
