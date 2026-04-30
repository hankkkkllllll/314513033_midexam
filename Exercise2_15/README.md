# CsiNet: Deep Learning-Based CSI Feedback & Compression

This repository contains the implementation of a Deep Learning-based Channel State Information (CSI) feedback system using CsiNet. It includes MATLAB scripts for generating wireless channel datasets via the COST2100 model and Python (Keras/TensorFlow) scripts for training a residual autoencoder to compress and reconstruct the CSI.

## 1. Theoretical Background: Evolution of CSI Feedback 

This project explores the DL-based approach to CSI feedback. Below is a comparison of the three generations of CSI feedback (Codebook, Compressed Sensing, and Deep Learning) across four critical dimensions:



---

## 2. Repository Overview

### Data Generation (`main_gen_data.m`)
The MATLAB script utilizes the **COST2100 channel model** to simulate an indoor wireless environment (`Indoor_CloselySpacedUser_2_6GHz` at $2.6$ GHz, LOS). 
* It generates **5 distinct datasets** (`ds1` to `ds5`).
* Random jitter and velocity offsets are applied to the Mobile Station (MS) positions across datasets to simulate dynamic channel conditions and test model generalizability.
* The raw channel impulse responses are transformed into the **angular-delay domain**, truncated to $32 \times 32$, and split into real and imaginary channels resulting in a shape of `(batch_size, 2, 32, 32)`.
* Data is split into Train (70%), Validation (15%), and Test (15%) sets.

### CsiNet Autoencoder (`CsiNet_train(b).py`or`CsiNet_train(c).py`)
The Python implementation uses Keras/TensorFlow to build **CsiNet**, an autoencoder with residual learning blocks.
* **Encoder:** Compresses the $32 \times 32 \times 2$ CSI matrix into a configurable `encoded_dim` (default is $512$, achieving a 1/4 compression ratio).
* **Decoder:** Decompresses the feature vector back to the original dimension using stacked $3 \times 3$ convolutional layers with residual connections (skip connections) to mitigate vanishing gradients and preserve fine channel details.
* **Loss Function:** Mean Squared Error (MSE).
* **Evaluation Metrics:** Normalized Mean Square Error (NMSE) and Cosine Correlation Coefficient ($\rho$).

There are two training strategies provided in the Python code:
1.  **Mixed Training:** Concatenates `ds1` to `ds5` into a single large training set to maximize data diversity.
2.  **Single Dataset Training (Generalization Test):** Trains *only* on `ds1`, but evaluates inference performance across `ds1` through `ds5` to test the robustness of the trained encoder/decoder against unobserved user positions/velocities.

## Environment Setup (Windows/RTX 4070)
The project is configured for Windows 11 with an NVIDIA RTX 4070 GPU :
1. **Create the virtual environment with the specific Python version:**
    ```bash
    conda create -n tensorflow python=3.9.19 -y

2. **Activate the environment:**
    ```bash
    conda activate tensorflow

3. **Install CUDA Toolkit via Conda:**
    ```bash
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

4. **Install all required packages via requirements.txt:**
   ```bash
   pip install -r requirements.txt
## 4. How to Run

### Step 1: Generate Datasets
Run the MATLAB script. It will create a `mydata/` directory and populate it with `.mat` files containing the training, validation, and testing matrices for all 5 environments.
```matlab
% In MATLAB command window
run('generate_data.m')
```
### Step 2: Training Model
To train on ds1 only and evaluate generalization on ds1 to ds5
```python
run('CsiNet_train(b).py')
```
### Step 3: Training Model (compare)
To train the model using the mixed dataset approach
```python
run('CsiNet_train(c).py')
```