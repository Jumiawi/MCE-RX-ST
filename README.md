# MCE-RX-ST: A Hybrid Deep Learning Model for Visual Classification in Remote Sensing Using Minimum Cross Entropy, ResNeXt, and Swin Transformer

This repository contains the implementation of **MCE-RX-ST**, a hybrid deep learning model that integrates **Minimum Cross Entropy Thresholding (MCET)**, **ResNeXt CNN**, and **Swin Transformer** for remote sensing image classification.

## Journal Submission
This research has been submitted for peer review to **The Visual Computer** journal under the title:

**MCE-RX-ST: A Hybrid Deep Learning Model for Visual Classification in Remote Sensing Using Minimum Cross Entropy, ResNeXt, and Swin Transformer**

## Overview
This model leverages:
- **MCET-based segmentation masks** (using histogram skewness and Gumbel distribution-based mean estimation)
- **Deep feature extraction using Swin Transformer and ResNeXt CNN**
- **Feature fusion combining deep features with MCET masks**

## Repository Structure
```
│── main.m                    % Main script to run the hybrid model
│── load_datasets.m            % Load WHU-RS19 and UCMerced datasets
│── compute_MCET_masks.m       % Compute segmentation masks based on MCET
│── extract_swin_features.m    % Extract features using Swin Transformer
│── extract_resnext_features.m % Extract features using ResNeXt CNN
│── apply_histogram_skewness_check.m  % Determine skewness & mean estimation
│── fuse_feature_maps.m        % Fuse feature maps with MCET masks
│── train_classifier.m         % Train classifier on combined features
│── evaluate_model.m           % Evaluate classification performance
│── adaptiveGumbelMCET.m       % Apply MCET method based on histogram
│── mingumbelmle.m             % Gumbel minimum estimation function
│── maxgumbelmle.m             % Gumbel maximum estimation function
│── plot_confusion_matrix.m    % Function to visualize the confusion matrix
│── README.md                  % Documentation for the repository
```

## Datasets
This model is trained and evaluated on:
- **WHU-RS19 Dataset** (Remote sensing images covering 19 classes)
- **UCMerced Land Use Dataset** (21-class dataset of aerial images)
•	[WHU-RS19]( https://huggingface.co/datasets/jonathan-roberts1/WHU-RS19)
•	[UCMerced LandUse]( http://weegee.vision.ucmerced.edu/datasets/landuse.html)

## Installation & Usage
1. Clone the repository:
2. Run the main script:
   ```matlab
   main
   ```

## Model Evaluation
- Classification accuracy is computed after training.
- The confusion matrix is plotted to assess class-wise performance.

## Results

![Confusion Matrix1](https://raw.githubusercontent.com/your-username/your-repo/main/confusion_matrix1.png)
![Sample Image1](images/sample_image1.png)

![Confusion Matrix2](https://raw.githubusercontent.com/your-username/your-repo/main/confusion_matrix2.png)
![Sample Image2](images/sample_image2.png)

## Citation
If you use this code, please cite our paper once it is accepted for publication in **The Visual Computer** journal.

