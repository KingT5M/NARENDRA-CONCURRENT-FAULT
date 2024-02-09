# NARENDRA-CONCURRENT-FAULT
This is a deep learning model used for detecting and classifying faults in Concurrent-Fault scenarios.
It deploys a Denoising Auto-Encoder (DAE) for feature extraction, followed by Gated Recurrent Unit (GRU) for feature learning.
It implements a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) architecture.

The architecture parameters of the proposed model.
    Layer Size and Parameters 
    CNN Layer 1 Filters: 8, Kernel Size: 2, Activation: ReLU, Input Shape: [30 × 2], Output Shape: [30 × 8] 
    CNN Layer 2 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8] 
    CNN Layer 3 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8] 
    CNN Layer 4 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8] 
    CNN Layer 5 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8] 
    Batch Normalization Layer 1 Output Shape: [30 × 8] 
    Max Pooling Layer Pool Size: 2, Output Shape: [15 × 8] 
    LSTM Layer 1 Neurons: 64, Activation: ReLU, Output Shape: [15 × 64] 
    LSTM Layer 2 Neurons: 64, Activation: ReLU, Output Shape: [15 × 64] 
    LSTM Layer 3 Neurons: 64, Activation: ReLU, Output Shape: [15 × 64] 
    LSTM Layer 4 Neurons: 64, Activation: ReLU, Output Shape: 64 
    Batch Normalization Layer 2 Output Shape: 64 
    Flatten Layer Output Shape: 64 
    Output Layer Output Shape: 9 


It also incorporates automatic hyper parameter tuning within the following ranges
Hyper-parameter ranges and values: 
    CNN Layers: 0–5 
    LSTM Layers: 0–5 
    Dense Layers: 0–5 
    Epochs: 50–900 
    Max Pooling Layer: 0–1 
    Drop Layer: 0–2 
    Batch Normalization Layer: 0–2 
    Batch Size: 64–150 
    Learning Rate: 0.001–0.0001

The author is - Ian Mwaniki Kanyi alias King T5M. ALL HAIL T5M.
