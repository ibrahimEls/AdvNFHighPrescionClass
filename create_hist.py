import json
import os
import numpy as np
import torch
import argparse
import pandas as pd

# Import custom functions and models.
from lib.data_utils import return1j2j, load_nf_models
from lib.class_model import CombinedClassifier

def createJetData(jet_num, useTestData, set_mu=3, seed=0, n_param=[1, 1, 1, 1, 1, 0], useRand=False,data=[]):
    """
    Create jet data with optional systematic variations and data processing.

    Parameters:
        jet_num (int or str): Jet number to filter (or "all" to return full dataset).
        useTestData (bool): Whether to use test data.
        set_mu (int, optional): Mu parameter for bootstrapping. Defaults to 3.
        seed (int, optional): Random seed. Defaults to 0.
        n_param (list, optional): List of systematic parameters. Defaults to [1,1,1,1,1,0].
        useRand (bool, optional): Whether to apply random systematic shifts. Defaults to False.

    Returns:
        tuple: Processed data tensor, label tensor, weights, and feature names.
    """


    # Optionally apply random systematic shifts
    if useRand:
        random_state = np.random.RandomState(seed)
        print("applying systmatics")
        n_param[-3] = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
        n_param[-2] = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
        n_param[-1] = np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)
        print(n_param)

    # Import functions for bootstrapping and systematics
    from systematics import get_bootstrapped_dataset, get_systematics_dataset

    # Get the test set (assumed to be defined in a global 'data' object)
    test_set = data.get_test_set()

    # Create a pseudo-experimental dataset with bootstrapping
    pseudo_exp_data = get_bootstrapped_dataset(
        test_set,
        mu=set_mu,
        ttbar_scale=n_param[0],
        diboson_scale=n_param[1],
        bkg_scale=n_param[2],
        seed=seed,
        get_ans=True
    )

    # Prepare weights and detailed labels
    weights = np.ones(pseudo_exp_data.shape[0])
    detailed_labels = pseudo_exp_data["Label"]
    pseudo_exp_data.drop(columns="Label", inplace=True)
    labels = detailed_labels[detailed_labels == "htautau"]

    # Apply systematics to the pseudo-experimental data
    data_vis = systematics(
        data_set={
            "data": pseudo_exp_data,
            "weights": weights,
            "detailed_labels": detailed_labels,
            "labels": labels
        },
        tes=n_param[3],
        jes=n_param[4],
        soft_met=n_param[5],
    )

    # If jet_num is not "all", filter by jet number
    if jet_num != "all":
        filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(jet_num, data_vis)
        temp_labels = (filtered_det_labels.values == "htautau")
        temp_labels = torch.tensor([int(val) for val in temp_labels])
    else:
        return data_vis, detailed_labels

    if not useTestData:
        # Compute background ratios relative to non-signal events
        ratio_ztt = len(filtered_data[filtered_det_labels == "ztautau"]) / len(filtered_data[filtered_det_labels != "htautau"])
        ratio_ttbar = len(filtered_data[filtered_det_labels == "ttbar"]) / len(filtered_data[filtered_det_labels != "htautau"])
        ratio_diboson = len(filtered_data[filtered_det_labels == "diboson"]) / len(filtered_data[filtered_det_labels != "htautau"])

        # Get the training set and limit the number of events
        data_vis_train = data.get_train_set()
        MAX_NUM_EVENTS = 5000000
        for key in data_vis_train.keys():
            if key != "settings":
                try:
                    subset = data_vis_train[key]
                    subset = subset.iloc[:MAX_NUM_EVENTS].reset_index(drop=True)
                    data_vis_train[key] = subset
                except:
                    data_vis_train[key] = data_vis_train[key][:MAX_NUM_EVENTS]

        # Apply systematics to the training data
        data_vis = systematics(
            data_set=data_vis_train,
            tes=n_param[3],
            jes=n_param[4],
            soft_met=n_param[5],
        )

        filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(jet_num, data_vis)

        # Determine event counts based on computed ratios
        count_ztt = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ztt)
        count_ttbar = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ttbar)
        count_diboson = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_diboson)

        # Create balanced datasets for signal and backgrounds
        temp_labels = []
        signal_data = filtered_data[filtered_det_labels == "htautau"]
        temp_labels.extend([1] * len(signal_data))
        ztt_data = filtered_data[filtered_det_labels == "ztautau"][:count_ztt]
        temp_labels.extend([0] * len(ztt_data))
        ttbar_data = filtered_data[filtered_det_labels == "ttbar"][:count_ttbar]
        temp_labels.extend([0] * len(ttbar_data))
        diboson_data = filtered_data[filtered_det_labels == "diboson"][:count_diboson]
        temp_labels.extend([0] * len(diboson_data))

        # Concatenate the subsets
        filtered_data = pd.concat((signal_data, ztt_data, ttbar_data, diboson_data), ignore_index=True)

    # Convert to torch tensors
    filtered_data = torch.tensor(filtered_data.values)
    filtered_det_labels = torch.tensor(temp_labels)

    # Remove any rows with a value of -25
    mask = torch.any(filtered_data == -25, dim=1)
    filtered_data = filtered_data[~mask]
    filtered_det_labels = filtered_det_labels[~mask]

    # Determine columns on which to apply logarithm transform based on jet type
    if jet_num == 1:
        log_columns = [0, 3, 6, 9, 10, 13, 14, 16, 17]
    elif jet_num == 0:
        log_columns = [0, 3, 6, 9, 10, 12, 13]
    else:
        log_columns = [0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23]

    for col_idx in range(filtered_data.shape[1]):
        if col_idx in log_columns:
            filtered_data[:, col_idx] = torch.log(filtered_data[:, col_idx])

    return filtered_data, filtered_det_labels, filtered_weights, feature_names

def main(args,data):
    """
    Main function to process jet data, generate histograms using a classifier model,
    and save the results to a JSON file.
    """
    # Load the classifier model from checkpoint.
    # The loaded model is expected to be callable with (data, jet_type) arguments.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    class_model_load = CombinedClassifier.load_from_checkpoint(args.class_model_path).to(device).eval().to(torch.float32)
    models = load_nf_models(args.models_dir,device)

    # Define the parameter arrays for jet energy scale (jes_arr) and testing scale (tes_arr).
    jes_arr = np.linspace(0.9, 1.1, 10)
    tes_arr = np.linspace(0.9, 1.1, 10)
    
    # Dictionary to store the histograms for each parameter combination.
    hist_dict_class = {}
    
    # Define histogram parameters.
    nbins = 200
    bins = np.linspace(0, 1, num=nbins)
    
    # Loop over combinations of test and jet energy scale parameters.
    for j in tes_arr:
        for i in jes_arr:
            # Define parameter list for data generation.
            n_params = [1, 1, 1, j, i, 0]
            
            # Create jet data using the provided root directory.
            alljet_data, _ = createJetData("all", True, set_mu=1000, seed=0, n_param=n_params, useRand=False, data=data)
            
            # Split the data into 2-jet and 1-jet sets and obtain corresponding labels.
            data_2j, data_1j, label_2j, label_1j = return1j2j(alljet_data,models)
            
            # Obtain classifier scores for each jet type without computing gradients.
            with torch.no_grad():
                scores_2j = torch.sigmoid(class_model_load(data_2j, 2)).cpu().numpy()
                scores_1j = torch.sigmoid(class_model_load(data_1j, 1)).cpu().numpy()
            
            # Concatenate scores and labels from both jet types.
            total_score = np.concatenate([scores_2j, scores_1j])
            total_label = np.concatenate([label_2j.numpy(), label_1j.numpy()])
            
            # Compute histograms for signal (label==1) and background (label==0) separately.
            S_hist_class, _ = np.histogram(total_score[total_label == 1], bins=bins, density=True)
            BG_hist_class, _ = np.histogram(total_score[total_label == 0], bins=bins, density=True)
            
            # Save the histograms in the dictionary keyed by the parameter tuple.
            hist_dict_class[(i, j)] = [S_hist_class, BG_hist_class]
    
    # Create a serializable dictionary to save as JSON.
    # Convert NumPy arrays to lists for JSON compatibility.
    serializable_dict = {
        str(key): {'sig': val[0].tolist(), 'bg': val[1].tolist()}
        for key, val in hist_dict_class.items()
    }
    
    # Save the dictionary to a JSON file using the provided file path.
    with open(args.json_save_path+"hist.json", 'w') as f:
        json.dump(serializable_dict, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Process jet data and generate histograms using a classifier model."
    )
    # Argument for the root directory containing data.
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="/Users/ibrahim/HEP-Challenge/",
        help="Root directory path for data files."
    )
    # Argument for the checkpoint path to load the classifier model.
    parser.add_argument(
        "--class_model_path", 
        type=str, 
        default="PreTrained/Models/DNN/DNNclass4NF_2.ckpt",
        help="Path to load classifier model checkpoint using CombinedClassifier.load_from_checkpoint."
    )
    # Argument for the output JSON file path.
    parser.add_argument(
        "--json_save_path", 
        type=str, 
        default="Test/SavedStats/",
        help="Path to save the resulting JSON file."
    )

    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="PreTrained/Models/",
        help="Path to the directory containing the '1_jet' and '2_jet' subdirectories with checkpoint files."
    )
    
    args = parser.parse_args()

    input_dir = os.path.join(args.root_dir, "input_data")
    program_dir = os.path.join(args.root_dir, "ingestion_program")
    score_dir = os.path.join(args.root_dir, "scoring_program")
    # Append directories so that modules from these paths can be imported
    import sys
    sys.path.append(program_dir)
    sys.path.append(score_dir)

    # Import the required functions from the ingestion program
    from systematics import systematics
    from datasets import Data  # Data class for non-public dataset

    use_public_dataset = True
    if use_public_dataset:
        from datasets import Neurips2024_public_dataset as public_dataset
        data = public_dataset()
    else:
        data = Data(input_dir)

    print("Loading Data")
    # Load train and test sets
    data.load_train_set()
    data.load_test_set()

    main(args,data)

