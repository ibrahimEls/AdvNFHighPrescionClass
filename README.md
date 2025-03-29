# Adversarial Normalizing Flows for High Precision Classification
## Solution to the FAIR UNIVERSE - Higgs Uncertainty Challenge
## Author: Ibrahim Elsharkawy

This document describes the hardware setup, software dependencies, installation steps, and instructions for training, evaluating, and predicting with the HEP-Challenge models. Please read carefully to ensure the correct environment and file structure.

---

## 1. Hardware Setup

### 1.1 Hardware Used for Development
- **Machine:** MacBook Pro (production)
- **Processor:** Apple M3 Max (unified CPU and GPU)
- **Memory:** 64 GB RAM
- **GPU:** Integrated Apple GPU (unified architecture)

### 1.2 Minimum Hardware Configuration for Deployment
- **Memory:** 64 GB RAM
- **GPU:** 1 NVIDIA T4 GPU

---

## 2. Operating System / Platform
- **OS:** macOS Sonoma 14.1

---

## 3. Software Dependencies and Installation

### 3.1 Required Software
- **Python:** (Ensure you are using a compatible version, e.g., Python 3.8+)
- **pip:** For installing Python dependencies
- **3rd-Party Libraries:** Listed in `requirements.txt`  
- **Docker:** (Optional, see Docker section below)

### 3.2 Installation Steps

1. **Clone the HEP-Challenge Repository**

   The HEP-Challenge repository must be cloned *outside* the current archive. Replace `/desired/path/HEP-Challenge/` with your preferred directory.

   ```bash
   git clone https://github.com/FAIR-Universe/HEP-Challenge.git /desired/path/HEP-Challenge/
   ```

2. **Download Public Data**

   Download the public dataset with the following command:

   ```bash
   wget -O public_data.zip https://www.codabench.org/datasets/download/b9e59d0a-4db3-4da4-b1f8-3f609d1835b2/
   ```

3. **Move and Extract the Data**

   Move the downloaded zip file into the HEP-Challenge repository, then extract it:

   ```bash
   mv public_data.zip /desired/path/HEP-Challenge/
   cd /desired/path/HEP-Challenge/
   unzip public_data.zip
   ```

4. **Install Python Dependencies**

   From the root of this project, install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Model Training and Execution

### 4.1 Training the Normalizing Flow (NF) Model

Run `train_NF.py` with the following arguments:

```bash
python train_NF.py \
  --root-dir "/Users/ibrahim/HEP-Challenge/" \
  --checkpoint-path "Test/" \
  --c 1 \
  -s True
```

**Argument Details:**
- `--root-dir`: Root directory containing `input_data`, `ingestion_program`, and `scoring_program` folders.
- `--checkpoint-path`: Directory to save and/or load the model checkpoint.
- `--c`: Hyperparameter (default value is 1).
- `-s`: Boolean flag indicating whether to train on signal data.

---

### 4.2 Training the Classifier Model

Run `train_class.py` with these arguments:

```bash
python train_class.py \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --models_dir "PreTrained/Models/" \
  --checkpoint_path "Test/"
```

**Argument Details:**
- `--root_dir`: Root directory for data files.
- `--models_dir`: Directory containing subdirectories `1_jet` and `2_jet` with checkpoint files.
- `--checkpoint_path`: Directory where model checkpoints will be saved during training.

---

### 4.3 Creating Histograms

Run `create_hist.py` with the following arguments:

```bash
python create_hist.py \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --class_model_path "PreTrained/Models/DNN/DNNclass4NF_2.ckpt" \
  --json_save_path "Test/SavedStats/" \
  --models_dir "PreTrained/Models/"
```

**Argument Details:**
- `--root_dir`: Root directory path for data files.
- `--class_model_path`: Path to load the classifier model checkpoint.
- `--json_save_path`: Directory where the resulting JSON file will be saved.
- `--models_dir`: Directory containing NF model checkpoints.

---

### 4.4 Running Neyman Construction

Run `create_neyman.py` with these arguments:

```bash
python create_neyman.py \
  --hist_path "PreTrained/SavedStats/histogram_2jet1jet_class_4nf_200bins.json" \
  --json_save_path "Test/SavedStats/" \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --class_model_path "PreTrained/Models/DNN/DNNclass4NF_2.ckpt" \
  --models_dir "PreTrained/Models/"
```

**Argument Details:**
- `--hist_path`: Path to the input histogram JSON file.
- `--json_save_path`: Directory to save the output JSON file.
- `--root_dir`: Root directory path for data files.
- `--class_model_path`: Path to the classifier model checkpoint.
- `--models_dir`: Directory containing `1_jet` and `2_jet` subdirectories with NF model checkpoint files.

---

### 4.5 Prediction

Run `predict.py` with these arguments:

```bash
python predict.py \
  --hist_path "PreTrained/SavedStats/histogram_2jet1jet_class_4nf_200bins.json" \
  --json_save_path "Test/SavedStats/" \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --class_model_path "PreTrained/Models/DNN/DNNclass4NF_2.ckpt" \
  --models_dir "PreTrained/Models/" \
  --neyman_path "PreTrained/SavedStats/2jet_1jet_2NP_MLE_Final_DNN_test_4NF_200.json" \
  --mu 1 \
  --predict_numevents \
  --nevent 10
```

**Argument Details:**
- `--hist_path`: Path to the input histogram JSON file.
- `--json_save_path`: Directory to save the resulting JSON file.
- `--root_dir`: Root directory for data files.
- `--class_model_path`: Path to load the classifier model checkpoint.
- `--models_dir`: Directory containing NF model checkpoints.
- `--neyman_path`: Path to the Neyman JSON file.
- `--mu`: Hyperparameter value (default is 1).
- `--predict_numevents`: Flag to indicate whether to predict mu on a test dataset.
- `--nevent`: Number of events to test if `predict_numevents` is not activated.

---

## 5. Important Side Effects

- **Directory Structure:**  
  - Models are saved in `Test/lighting_logs`. The subsequent processing steps will fail if the directory structure is not maintained or if the paths are incorrect.
  
- **Data Overwrite:**  
  - The data processing scripts may overwrite original data if not handled with care. Always ensure backups if necessary.

---

## 6. Key Assumptions

- The HEP-Challenge repository is cloned outside the current archive and its path is correctly provided via the `--root-dir` argument.
- The repository includes the necessary folder structure (`input_data`, `ingestion_program`, `scoring_program`).
- Public data is downloaded, moved, and extracted into the HEP-Challenge repository as specified.
- The output directories (e.g., `Test/SavedStats/`) are empty or properly configured before starting a new training or evaluation run.

---

## 7. Docker (Optional)

A Dockerfile is provided for setting up the environment easily. This can be used as an alternative to manual dependency installation.

**Dockerfile:**

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["bash"]
```

**Usage:**

```bash
# Build the Docker image
docker build -t hep-challenge .

# Run the container
docker run -it hep-challenge
```

---

This README provides a concise yet detailed guide to setting up the hardware/software environment, installing dependencies, and executing the model training, evaluation, and prediction steps. Please ensure that all paths are adjusted according to your local setup.
```