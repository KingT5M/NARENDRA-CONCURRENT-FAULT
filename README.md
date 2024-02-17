#NARENDRA-CONCURRENT-FAULT
This is a deep reinforcement learning model used for detecting and classifying faults in Concurrent-Fault scenarios.
It deploys a Denoising Auto-Encoder (DAE) for feature extraction, followed by Gated Recurrent Unit (GRU) for feature learning.
It implements a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) architecture.

The architecture parameters of the proposed model.
    Layer Size and Parameters,
    CNN Layer 1 Filters: 8, Kernel Size: 2, Activation: ReLU, Input Shape: [30 × 2], Output Shape: [30 × 8],
    CNN Layer 2 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8],
    CNN Layer 3 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8],
    CNN Layer 4 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8],
    CNN Layer 5 Filters: 8, Kernel Size: 2, Activation: ReLU, Output Shape: [30 × 8],
    Batch Normalization Layer 1 Output Shape: [30 × 8],
    Max Pooling Layer Pool Size: 2, Output Shape: [15 × 8],
    LSTM Layer 1 Neurons: 64, Activation: ReLU, Output Shape: [15 × 64],
    LSTM Layer 2 Neurons: 64, Activation: ReLU, Output Shape: [15 × 64],
    LSTM Layer 3 Neurons: 64, Activation: ReLU, Output Shape: [15 × 64],
    LSTM Layer 4 Neurons: 64, Activation: ReLU, Output Shape: 64,
    Batch Normalization Layer 2 Output Shape: 64,
    Flatten Layer Output Shape: 64,
    Output Layer Output Shape: 9,

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



## Reinforcement Learning in the provided code:

The code implements **episodic Reinforcement Learning (RL)** with the **REINFORCE algorithm** to optimize hyperparameters for a fault detection model. Here's a breakdown of the key elements:

**Agent:** The agent is the script that samples, trains, and evaluates different model configurations based on hyperparameters.

**Environment:** The environment is the training process itself. The agent interacts with it by training models with different hyperparameters and receiving rewards based on their validation performance.

**State:** The state of the environment consists of the current hyperparameter configuration being evaluated.

**Action:** The action taken by the agent is to randomly sample a new set of hyperparameters based on the previous state and rewards.

**Reward:** The reward is calculated based on the validation accuracy of the model trained with the current hyperparameters. A higher reward is given for higher accuracy.

**REINFORCE Algorithm:**

1. **Episode Loop:** The code runs for a specified number of episodes.
2. **Step Loop:** In each episode, the agent takes steps within a limit.
3. **Sample Hyperparameters:** For each step, the agent randomly samples a new set of hyperparameters based on the current values and some noise.
4. **Build and Train Model:** The agent builds and trains a model using the sampled hyperparameters.
5. **Evaluate Model:** The model is evaluated on the validation set, and the accuracy is obtained.
6. **Calculate Reward:** The reward is calculated based on the accuracy, with a custom reward function (example: accuracy - 0.5).
7. **Update Hyperparameters:** The current hyperparameters are updated using REINFORCE, adjusting their values based on the received reward. This encourages exploring hyperparameters that lead to better rewards.
8. **Repeat Steps:** Steps 3-7 are repeated for the remaining steps in the episode.

**Hyperparameter Search:**

* The code defines ranges for various hyperparameters like number of layers, dropout rate, etc.
* It explores different combinations within these ranges during each episode.
* The best performing hyperparameters based on the final reward are identified after all episodes.

**Final Model Training:**

* The final model is trained with the best hyperparameters found during RL.
* The model is further evaluated on the test set, and metrics like precision, recall, and F1-score are calculated for each fault type.

**Overall,** the code demonstrates how RL can be effectively used to optimize hyperparameters for a machine learning model, leading to improved performance in this specific fault detection task.



The author is - Ian Mwaniki Kanyi alias King T5M. ALL HAIL T5M.
