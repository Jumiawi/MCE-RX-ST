% Define the path to your image folder
imageFolder = 'path_to_your_image_folder';

% Create an imageDatastore to read the images from the folder
imds = imageDatastore(imageFolder, 'FileExtensions', '.jpg', 'LabelSource', 'foldernames');

% Define the image augmentation options using imageDataAugmenter
imageAugmenter = imageDataAugmenter(...
    'RandRotation', [-30, 30], ...  % Random rotation between -30 and 30 degrees
    'RandFlip', 'horizontal', ...   % Random horizontal flip
    'RandScale', [0.8, 1.2]);       % Random scaling between 80% and 120%

% Apply the augmentations to the image datastore
augimds = augmentedImageDatastore([224 224], imds, 'DataAugmentation', imageAugmenter);

% Example: Display a few augmented images to inspect the augmentation
numImagesToDisplay = 5;
figure;
for i = 1:numImagesToDisplay
    subplot(1, numImagesToDisplay, i);
    img = read(augimds);  % Read an augmented image
    imshow(img);
end

% Now, you can use the augmented image datastore (augimds) for training your deep learning model
