# MCUa-Model

This repository is the part A of the ICIAR 2018 Grand Challenge on BreAst Cancer Histology (BACH) images for automatically classifying H&E stained breast histology microscopy images in four classes: normal, benign, in situ carcinoma and invasive carcinoma.


We propose a novel dynamic ensemble Convolutional Neural Network with terming Multi-level Context and Uncertainty aware (MCUa) model for the automated classification of H&E
stained breast histology images. First, we resize input images into two different scales to capture multi-scale local information. Then we designed patch feature extractor networks by extracting patches and feed them to pre-trained fine-tuned DCNNs (i.e. DenseNet-161 and ResNet-152). The extracted feature maps are then used by our context-aware
networks to extract multi-level contextual information from different pattern levels. Finally, a novel uncertainty-aware model ensembling stage is developed to dynamically select
the most certain context-aware models for the final prediction.

![MCUA_model](https://user-images.githubusercontent.com/20457990/107374459-85cd2f00-6adf-11eb-9356-f6a5202e8969.PNG)


## Citation
If you use this code for your research, please cite our paper: [MCUa: Multi-level Context and Uncertainty aware Dynamic Deep Ensemble for Breast Cancer Histology Image Classification](https://ieeexplore.ieee.org/document/9525263?denied=)



```
@ARTICLE{MCUA,
  author={Senousy, Zakaria and Abdelsamea, Mohammed and Gaber, Mohamed Medhat and Abdar, Moloud and Acharya, Rajendra U and Khosravi, Abbas and Nahavandi, Saeid},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={MCUa: Multi-level Context and Uncertainty aware Dynamic Deep Ensemble for Breast Cancer Histology Image Classification}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TBME.2021.3107446}}
```
