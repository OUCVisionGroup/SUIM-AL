<div align="center">
<h1>Enhance Underwater Imaging in Nonuniform Artificial Light: An Extended Benchmark Dataset and A Dual-Stream Co-training Enhancement Model

<h4 align="center">
    <a href="https://github.com/OUCVisionGroup/SUIM-AL" target='_blank'>[Project Page]</a> •
    <a href="https://ieeexplore.ieee.org/abstract/document/11429073" target='_blank'>[Paper]</a>
</h4>

</div>

---

## SUIM-AL dataset 
Based on the physical imaging model, we introduce an AL incorporated underwater imaging synthesis pipeline and SUIM-AL — a large-scale underwater image benchmark with 16350 samples across ten water types and diverse lighting conditions, including artificial light.
<div>
   <img src="./SUIM-AL.png" width="80%" alt="teaser" align=center />
</div>
The complete dataset as well as the training and testing sets used for the enhancement model have all been uploaded.[SUIM-AL dataset](https://drive.google.com/drive/home)

## DSFormer 
We propose DSFormer, a dual-stream Transformer network with shared weights, trained on pairs of images under varying lighting and degradation conditions to improve robustness.
<div>
   <img src="./DSFormer.png" width="80%" alt="teaser" align=center />
</div>

### 💻Setup
- PyTorch >= 1.11
- CUDA >= 11.3  
- For other dependencies and environment configurations, please refer to [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer).

### ▶️Model inference 
- Download the pretrained model and modify the configuration file path.

```
python Enhancement/test_from_dataset.py
```
### ▶️Model train 
- Download the CLIP prompt model and modify the configuration file.

```
python basicsr/train.py --opt Options/DSFormer.yml
```

## 🙏 Thanks
Our code is based on [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) and [CLIP-LIT](https://github.com/ZhexinLiang/CLIP-LIT). You can refer to their README files and source code for more implementation details.

The dataset SUIM-AL and source code and pre-trained model of the enhanced model DSFormer are about to be released.

## 📖 Citation

If you find our work useful, please consider citing:

```
@article{ge2026enhance,
  title={Enhance Underwater Imaging in Nonuniform Artificial Light: An Extended Benchmark Dataset and A Dual-Stream Co-training Enhancement Model},
  author={Ge, Zhou and Mei, Han and Li, Kunqian and Song, Dalei},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```
