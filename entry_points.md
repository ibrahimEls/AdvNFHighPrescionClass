# Entrypoints for Adversarial Normalizing Flows for High Precision Classification

This document lists the commands required to execute the various steps of the HEP-Challenge pipeline along with explicit explanations for each argument.

---

## Table of Contents
1. [Training the Normalizing Flow (NF) Model](#training-the-normalizing-flow-nf-model)
2. [Training the Classifier Model](#training-the-classifier-model)
3. [Creating Histograms](#creating-histograms)
4. [Running Neyman Construction](#running-neyman-construction)
5. [Prediction](#prediction)

---

## 1. Training the Normalizing Flow (NF) Model

Execute the following command to train the NF model:

```bash
python train_NF.py \
  --root-dir "/Users/ibrahim/HEP-Challenge/" \
  --checkpoint-path "Test/" \
  --c 1 \
  -s True
```

**Argument Explanations:**
- `--root-dir`:  
  Specifies the root directory containing the necessary folders (`input_data`, `ingestion_program`, and `scoring_program`).  
- `--checkpoint-path`:  
  Directory to save the NF model checkpoints (and optionally, load a pre-existing checkpoint).  
- `--c`:  
  A hyperparameter (default value is 1) used during training.  
- `-s`:  
  Boolean flag indicating whether to train using signal data (`True` means signal training is enabled).

---

## 2. Training the Classifier Model

Run the classifier training with the command below:

```bash
python train_class.py \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --models_dir "PreTrained/Models/" \
  --checkpoint_path "Test/"
```

**Argument Explanations:**
- `--root_dir`:  
  The root directory path for all data files.
- `--models_dir`:  
  Directory containing the subdirectories (`1_jet` and `2_jet`) that hold the NF model checkpoints.
- `--checkpoint_path`:  
  Directory where the classifier model checkpoints will be saved during training.

---

## 3. Creating Histograms

Generate histograms by running the following command:

```bash
python create_hist.py \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --class_model_path "PreTrained/Models/DNN/DNNclass4NF_2.ckpt" \
  --json_save_path "Test/SavedStats/" \
  --models_dir "PreTrained/Models/"
```

**Argument Explanations:**
- `--root_dir`:  
  Root directory path for data files.
- `--class_model_path`:  
  File path to load the classifier model checkpoint using `CombinedClassifier.load_from_checkpoint`.
- `--json_save_path`:  
  Directory where the output JSON file (histograms) will be saved.
- `--models_dir`:  
  Directory containing the NF model checkpoints organized in the subdirectories `1_jet` and `2_jet`.

---

## 4. Running Neyman Construction

Perform Neyman construction using the following command:

```bash
python create_neyman.py \
  --hist_path "PreTrained/SavedStats/histogram_2jet1jet_class_4nf_200bins.json" \
  --json_save_path "Test/SavedStats/" \
  --root_dir "/Users/ibrahim/HEP-Challenge/" \
  --class_model_path "PreTrained/Models/DNN/DNNclass4NF_2.ckpt" \
  --models_dir "PreTrained/Models/"
```

**Argument Explanations:**
- `--hist_path`:  
  Path to the input histogram JSON file.
- `--json_save_path`:  
  Directory where the output JSON file from the Neyman construction will be saved.
- `--root_dir`:  
  Root directory path for the data files.
- `--class_model_path`:  
  Path to load the classifier model checkpoint.
- `--models_dir`:  
  Directory containing the NF model checkpoints stored in `1_jet` and `2_jet` subdirectories.

---

## 5. Prediction

Run predictions with the command below:

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

**Argument Explanations:**
- `--hist_path`:  
  Path to the input histogram JSON file.
- `--json_save_path`:  
  Directory where the resulting JSON file from the prediction will be saved.
- `--root_dir`:  
  Root directory containing the data files.
- `--class_model_path`:  
  Path to load the classifier model checkpoint (used with `CombinedClassifier.load_from_checkpoint`).
- `--models_dir`:  
  Directory containing the NF model checkpoints in the `1_jet` and `2_jet` subdirectories.
- `--neyman_path`:  
  File path to the Neyman JSON file needed for prediction.
- `--mu`:  
  A hyperparameter (float) with a default value of 1, used in the prediction process.
- `--predict_numevents`:  
  A flag that, when set, instructs the script to predict the mu value on a test dataset.
- `--nevent`:  
  Specifies the number of events to test if the `--predict_numevents` flag is not activated.

---