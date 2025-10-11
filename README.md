WTSNet: An Accurate Stereo Matching Network Based on Wavelet Transform and Superpixel Segmentation**


WTSNet: An Accurate Stereo Matching Network Based on Wavelet Transform and Superpixel Segmentation**
Haiming Qu, Yunhui Luo*, Chongbao Zhao, Minxuan He, Qing Wang
State Key Laboratory of Green Papermaking and Resource Recycling, Qilu University of Technology (Shandong Academy of Sciences)
*Corresponding author: [Lyh@qlu.edu.cn](mailto:Lyh@qlu.edu.cn)

📘 Paper link: [https://github.com/Qhm135/WTSNet](https://github.com/Qhm135/WTSNet)

---

🧠 Introduction

WTSNet is a high-accuracy stereo matching network designed to improve disparity estimation by jointly leveraging wavelet-based multi-frequency analysis and superpixel-based structural guidance.

The Wavelet Transform-based Module (WTM) decomposes images into four frequency sub-bands (LL, LH, HL, HH) with learnable filters and sub-band attention, enhancing both global and local feature representation.
The Superpixel Segmentation Module (SSM) generates edge probability maps to guide disparity estimation along structural boundaries, improving edge precision.
The network integrates multi-scale cost volume construction, 3D hourglass aggregation, and cross-scale fusion to produce accurate and smooth disparity maps.

WTSNet achieves 0.45 px EPE on KITTI 2012 and 0.73 px EPE on KITTI 2015, outperforming several strong baselines including TANet and PSMNet.


🏗️ Architecture Overview

The network consists of the following modules:

1. Multi-Scale Feature Extraction – Extracts local and global features using ResNet + FPN.
2. Wavelet Transform-based Attention Module (WTM) – Learns frequency-domain representations with sub-band attention.
3. Superpixel Segmentation Module (SSM) – Generates structural edge maps to refine disparity estimation.
4. Multi-Scale Cost Volume Construction & Aggregation – Builds and fuses cost volumes across scales.
5. Cross-Scale Fusion & Disparity Regression – Produces the final dense disparity map.

<p align="center">
  <img src="https://github.com/Qhm135/WTSNet/blob/main/docs/architecture.png?raw=true" width="600">
</p>

---

⚙️ Environment Setup

Dependencies

* Python >= 3.8
* PyTorch >= 2.0
* torchvision >= 0.15
* CUDA >= 11.7
* OpenCV, NumPy, Matplotlib


🚀 Training

To train WTSNet on the KITTI Stereo dataset:

```bash
python train.py \
    --maxdisp 192 \
    --datapath /path/to/KITTI/ \
    --dataset kitti2015 \
    --epochs 300 \
    --batch_size 4 \
    --savemodel ./checkpoints/ \
    --logdir ./logs/ \
    --seed 1
```

Notes

For KITTI 2012, replace `--dataset kitti2015` with `kitti2012`.
Pretrained models can be specified via `--loadmodel path/to/model.pth`.


🧩 Testing

Evaluate a trained model on KITTI 2015 test images:

```bash
python test.py \
    --maxdisp 192 \
    --datapath /path/to/KITTI/ \
    --loadmodel ./checkpoints/wtsnet_best.pth \
    --pred_disp ./results/ \
    --error_vis ./visualizations/ \
    --no-cuda 0
```

---

📊 Performance

| Dataset    | EPE (px) | D1-error (%) | 3px-all (%) |
| ---------- | -------- | ------------ | ----------- |
| KITTI 2012 | **0.45** | 1.06         | 1.06        |
| KITTI 2015 | **0.73** | 2.40         | 2.40        |


📚 Citation

If you find this work useful, please cite:

```bibtex
@article{qu2025wtsnet,
  title={WTSNet: An Accurate Stereo Matching Network Based on Wavelet Transform and Superpixel Segmentation},
  author={Qu, Haiming and Luo, Yunhui and Zhao, Chongbao and He, Minxuan and Wang, Qing},
  journal={Under Review},
  year={2025},
  url={https://github.com/Qhm135/WTSNet}
}
```

---

## 📬 Contact

For questions or implementation issues, please contact:

* **Haiming Qu** – [zsang0960@gmail.com](mailto:zsang0960@gmail.com)
* **Yunhui Luo** – [Lyh@qlu.edu.cn](mailto:Lyh@qlu.edu.cn)
