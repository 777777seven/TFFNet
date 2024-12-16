# TFFNet: Texture-based Feature Fusion and Multi-scale Consistency for Deepfake Detection


## Datasets
FF++ dataset can download on [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)

If you use the FaceForensics++ data or code please cite:
```
@inproceedings{roessler2019faceforensicspp,
	author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
	title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
	booktitle= {International Conference on Computer Vision (ICCV)},
	year = {2019}
}

```


# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* TensorBoardX == 2.6.22
* Python == 3.10.14
* numpy == 1.26.4
* opencv-python == 4.10.0.84
* scipy == 1.14.0

## Thanks
We based our algorithm on the CORE code, [https//github.com/niyunsheng/CORE](https://github.com/niyunsheng/CORE)

```
@InProceedings{Ni_2022_CVPR,
    author    = {Ni, Yunsheng and Meng, Depu and Yu, Changqian and Quan, Chengbin and Ren, Dongchun and Zhao, Youjian},
    title     = {CORE: COnsistent REpresentation Learning for Face Forgery Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {12-21}
}
```
