# MCUa-Model

This repository is the part A of the ICIAR 2018 Grand Challenge on BreAst Cancer Histology (BACH) images for automatically classifying H&E stained breast histology microscopy images in four classes: normal, benign, in situ carcinoma and invasive carcinoma.


We propose a novel dynamic ensemble Convolutional Neural Network with terming Multi-level Context and Uncertainty aware (MCUa) model for the automated classification of H&E
stained breast histology images. First, we resize input images into two different scales to capture multi-scale local information. Then we designed patch feature extractor networks by extracting patches and feed them to pre-trained fine-tuned DCNNs (i.e. DenseNet-161 and ResNet-152). The extracted feature maps are then used by our context-aware
networks to extract multi-level contextual information from different pattern levels. Finally, a novel uncertainty-aware model ensembling stage is developed to dynamically select
the most certain context-aware models for the final prediction.

![MCUA MODEL](https://user-images.githubusercontent.com/20457990/101354531-396e3480-388d-11eb-9c12-b94584f2a3e6.PNG)
