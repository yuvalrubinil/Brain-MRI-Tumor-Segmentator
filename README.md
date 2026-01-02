# TransUNet - Brain MRI Tumor Segmentator
![](models/brain_tumor_model_v1/samples/sample3.png)

This project focuses on automated tumor segmentation in MRI brain scans using deep learning. Manual segmentation is time consuming and prone to variability, motivating the need for reliable automated solutions. The project aims to improve consistency and efficiency in medical image analysis, by reducing doctors workload and provide assistance in tumor detection.

### Architecture
The project applies a hybrid U-Net and Transformer architecture to combine local feature extraction with global contextual understanding. The model follows a U-Net structure with a convolutional encoder, a Transformer-based bottleneck, and a decoder for precise segmentation.

#### Encoder (CNN): 
Uses standard convolutional layers to extract local features (textures, edges) and progressively reduce spatial resolution.

#### Transformer Bottleneck:
Replaces the traditional convolutional bottleneck. It flattens feature maps into patches and applies self attention. This allows the model to capture global context and understand the relationship between distant pixels, which is essential for identifying the full extent of larger tumors.

#### Decoder (Transposed Conv):
Upsamples the features using nn.ConvTranspose2d to restore spatial resolution. It uses residual connections to bring high-resolution local details from the encoder back into the mask generation process.

![](architecture/architechture_white.png)

#### Activations
- **ReLU:** used in all convolutional blocks for non-linearity.
- **GELU:** used within the Transformer encoder to provide smooth non-linear activation suited for attention.
- **Sigmoid:** applied to the final output to convert the model’s predictions into a binary tumor segmentation mask.

### Data Collection/Preparation
We used the custom PyTorch dataset class - the MRIDataset - to load and preprocess MRI images together with their ground-truth images. To do it, we created an array called ‘samples’ and the pairs in it are stored as [MRI Scan, Ground Truth]. Only image–mask pairs for which both files exist, are included. 
All images and masks are resized to a fixed resolution of 256x256 pixels to ensure consistent input dimensions for the neural network. MRI images are resized and converted to tensors, while masks are resized using nearest-neighbor interpolation to preserve discrete label values and avoid the introduction of fake pixels that did not exist in the original image, created by the resizing process. Both images and masks are converted to grayscale, because there is no physical meaning for the color channel on this task. We want to get rid of redundant information to get better training time and less memory usage.

### Training and Evaluation
tbd...

### Analysis
tbd...


