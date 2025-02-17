%% MCE-RX-ST: A Hybrid Deep Learning Model for Visual Classification in Remote Sensing
% Walaa Jumiawi 
% This code integrates MCET segmentation, Swin Transformer, and ResNeXt

clc; clear; close all;

%% Load and Preprocess Data
% Define dataset directories
whu_dir = 'path_to_WHURS19';
ucm_dir = 'path_to_UCMerced';

% Load images and labels from datasets
[images, labels] = load_datasets(whu_dir, ucm_dir);

%% MCET-Based Segmentation Mask Extraction
masks = compute_MCET_masks(images);

%% Feature Extraction Using Swin Transformer
swin_features = extract_swin_features(images);

%% Feature Extraction Using ResNeXt CNN
resnext_features = extract_resnext_features(images);

%% Fusion of Feature Maps with Segmentation Masks
combined_features = fuse_feature_maps(swin_features, resnext_features, masks, 0.5);

%% Train Classification Model
[model, accuracy] = train_classifier(combined_features, labels);

disp('Classification Accuracy:');
disp(accuracy);

%% Evaluate Classification Performance
pred_labels = predict(model, combined_features);
conf_matrix = confusionmat(labels, pred_labels);
plot_confusion_matrix(conf_matrix);

%% Function Calls (Each Function in a Separate File)
function [images, labels] = load_datasets(whu_dir, ucm_dir)
    [images, labels] = load_datasets_function(whu_dir, ucm_dir);
end

function masks = compute_MCET_masks(images)
    masks = compute_MCET_masks_function(images);
end

function features = extract_swin_features(images)
    features = extract_swin_features_function(images);
end

function features = extract_resnext_features(images)
    features = extract_resnext_features_function(images);
end

function combined_features = fuse_feature_maps(swin_features, resnext_features, mcet_masks, alpha)
    combined_features = fuse_feature_maps_function(swin_features, resnext_features, mcet_masks, alpha);
end

function [model, accuracy] = train_classifier(features, labels)
    [model, accuracy] = train_classifier_function(features, labels);
end

function plot_confusion_matrix(conf_matrix)
    plot_confusion_matrix_function(conf_matrix);
end
