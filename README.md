# Brain MRI Tumor Segmentator
![](models/brain_tumor_model_v6/figures/poster.png)

This project focuses on automated tumor segmentation in MRI brain scans using deep learning. Manual segmentation is time consuming and prone to variability, motivating the need for reliable automated solutions. The project aims to improve consistency and efficiency in medical image analysis, by reducing doctors workload and provide assistance in tumor detection.

### Architecture
The project applies a hybrid U-Net and Transformer architecture to combine local feature extraction with global contextual understanding. The model follows a U-Net structure with a convolutional encoder, a Transformer-based bottleneck, and a decoder for precise segmentation.

#### Encoder: 
Uses standard convolutional layers to extract local features (textures, edges) and progressively reduce spatial resolution.

#### Transformer Bottleneck:
Replaces the traditional convolutional bottleneck. It flattens feature maps into patches and applies self attention. This allows the model to capture global context and understand the relationship between distant pixels, which is essential for identifying the full extent of larger tumors.

#### Decoder:
Upsamples the features using `nn.ConvTranspose2d` to restore spatial resolution. It uses residual connections to bring high-resolution local details from the encoder back into the mask generation process.

```mermaid 
graph LR

    %% Skip Connections
    E1 -.-> U3
    E2 -.-> U2
    E3 -.-> U1

    I["MRI - grayscale<br/>(1, 256, 256)"]
    I --> E1

    %% Encoder
    subgraph Encoder ["Encoder"]
        E1["Enc1<br/>(64, 256, 256)"]
        E2["Enc2<br/>(128, 128, 128)"]
        E3["Enc3<br/>(256, 64, 64)"]
        E4["Enc4<br/>(512, 32, 32)"]

        E1 --> E2
        E2 --> E3
        E3 --> E4
    end

    %% Bottleneck
    E4 --> T["Transformer Encoder<br/>4 layers, 8 heads<br/>(512, 32, 32)"]
    T --> U1

    %% Decoder
    subgraph Decoder ["Decoder"]
        U1["Up1<br/>(256, 64, 64)"]
        U2["Up2<br/>(128, 128, 128)"]
        U3["Up3<br/>(64, 256, 256)"]

        U1 --> U2
        U2 --> U3
    end

    U3 --> Logits
    Logits["Logits Heatmap<br/>(1, 256, 256)"]
    Logits --> O
    O["Binary Pred Mask<br/>(1, 256, 256)"]


    %% STYLING 
    style Encoder fill:transparent,stroke:#888,stroke-dasharray:0 50
    style Decoder fill:transparent,stroke:#888,stroke-dasharray:0 50

    classDef input fill:#424242,stroke:#616161,color:#FAFAFA
    classDef encoder fill:#1976D2,stroke:#1976D2,color:#FAFAFA
    classDef bottleneck fill:#7E57C2,stroke:#7E57C2,color:#FAFAFA
    classDef decoder fill:#B71C1C,stroke:#B71C1C,color:#FAFAFA
    classDef logits fill:#660000,stroke:#616161,color:#FAFAFA
    classDef output fill:#212121,stroke:#616161,color:#FAFAFA

    class I input
    class E1,E2,E3,E4 encoder
    class T bottleneck
    class U1,U2,U3 decoder
    class Logits logits
    class O output

    linkStyle 3 stroke:#8D6E63,stroke-width:3px
    linkStyle 4,5,6 stroke:#795548,stroke-width:3px
    linkStyle 7 stroke-width:2px
    linkStyle 8,9,10 stroke:#689F38,stroke-width:3px
    linkStyle 11 stroke:#F9A825,stroke-width:3px
    linkStyle 12 stroke:#303F9F,stroke-width:3px
```
- ![#FFFFFF](https://placehold.co/15x15/FFFFFF/FFFFFF.png) *Concatenate* - white dashed arrow
- ![#8D6E63](https://placehold.co/15x15/8D6E63/8D6E63.png) *Convolution (no pooling)*
- ![#795548](https://placehold.co/15x15/795548/795548.png) *Convolution*
- ![#F9A825](https://placehold.co/15x15/F9A825/F9A825.png) *Channel Projection*
- ![#689F38](https://placehold.co/15x15/689F38/689F38.png) *Transposed Convolution*
- ![#303F9F](https://placehold.co/15x15/303F9F/303F9F.png) *Sigmoid Threshold*


- architecture code: [`src/tumor_detector.py`](src/tumor_detector.py)

#### Activations
- **ReLU:** used in all convolutional blocks for non-linearity.
- **GELU:** used within the Transformer encoder to provide smooth non-linear activation suited for attention.
- **Sigmoid:** applied to the final logits to map raw values into a $[0, 1]$ probability range. These values are then binarized via thresholding to produce the final discrete tumor segmentation mask.

### Data Collection/Preparation
We used the custom PyTorch dataset class - the MRIDataset - to load and preprocess MRI images together with their ground-truth images. To do it, we created an array called ‘samples’ and the pairs in it are stored as [MRI Scan, Ground Truth]. Only image–mask pairs for which both files exist, are included. 
All images and masks are resized to a fixed resolution of 256x256 pixels to ensure consistent input dimensions for the neural network. MRI images are resized and converted to tensors, while masks are resized using nearest-neighbor interpolation to preserve discrete label values and avoid the introduction of fake pixels that did not exist in the original image, created by the resizing process. Both images and masks are converted to grayscale, because there is no physical meaning for the color channel on this task. We want to get rid of redundant information to get better training time and less memory usage.

- dataset code: [`src/dataset.py`](src/dataset.py)

### Training and Evaluation
We did not immediately train on the full dataset. Instead, we began with a small pilot subset of 20 patients to observe the model’s behavior. During this phase, we focused on tuning two main axes: the loss function and the learning rate.
#### Loss Function:
We explored various linear combinations of Dice and BCE loss. We initially hypothesized that prioritizing the Dice component ($\mathcal{L} = 0.8 \cdot \mathcal{dice} + 0.2 \cdot \mathcal{BCE}$) would improve mask coherence. however, this setup sometimes produced blobby or over smoothed masks. Through iterative testing, we shifted the focus and gave more weight to the BCE and determined that a balanced combination:

$$
\mathcal{L} = 0.5 \cdot \mathcal{L}_{Dice} + 0.5 \cdot \mathcal{L}_{BCE}
$$

yielded the most stable results. By balance, we achieved good overlap and good pixel confidence.

#### Learning Rate:
Initially, we trained the model using a fixed learning rate. We then used a learning rate scheduler (ReduceLROnPlateau) to monitor the average validation Dice score, automatically reducing the learning rate when performance plateaued, which enabled finer weight updates and better convergence.

![](models/brain_tumor_model_v6/figures/training_curve.png)
- training code: [`models/brain_tumor_model_v6/notebooks/training.ipynb`](models/brain_tumor_model_v6/notebooks/training.ipynb)

### Analysis
#### Computational Challenge
Since our architecture is not a standard U-Net and includes an attention based Transformer block, training was significantly more computationally demanding.
Initially, we used only three encoder blocks, reducing the feature map resolution to **64 × 64**. These large feature maps created a severe bottleneck at the Transformer stage, exhausting available VRAM and eventually causing system crashes.
To address this, we added a fourth encoder block (`enc4`), reducing the feature map resolution to **32 × 32**, which fit within our VRAM constraints (see the architecture diagram for clarity).

#### Results
We successfully trained a robust model (**`brain_tumor_model_v6`**) that achieves an average Dice score of **88%+** and **92%+** Recall. The predictions are coherent and stable across most inputs.

***Performance Metrics:***

| Metric     | Value                      |
|------------|----------------------------|
| Avg Dice   | 0.8829                     |
| Accuracy   | 0.9184                     |
| Precision  | 0.8417                     |
| Recall     | 0.9213                     |
| F1-score   | 0.8797                     |
| TP / TN / FP / FN | 0.3 / 0.62 / 0.055 / 0.025 |

- View sample predictions: [`models/brain_tumor_model_v6/figures/samples.png`](models/brain_tumor_model_v6/figures/samples.png)  
- Run a full evaluation: [`models/brain_tumor_model_v6/notebooks/evaluation.ipynb`](models/brain_tumor_model_v6/notebooks/evaluation.ipynb)
- Metrics code: [`src/metrics.py`](src/metrics.py)

