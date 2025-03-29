import torch
import pandas as pd
import numpy as np
import os
from .NormFlow import NormalizingFlowModel

def filterbyjet(jet_num, data_vis):
    """
    Filter the dataset based on the number of jets.

    Parameters:
        jet_num (int): The jet count to filter on (0, 1, or 2).
        data_vis (dict): Dictionary with keys "data", "detailed_labels", "weights", "labels".

    Returns:
        tuple: (filtered_data, filtered_det_labels, filtered_weights, feature_names)
    """
    if jet_num == 2:
        # Filter rows with PRI_n_jets >= 2
        filtered_data = data_vis["data"][data_vis["data"]["PRI_n_jets"] >= jet_num]
        filtered_det_labels = data_vis["detailed_labels"][data_vis["data"]["PRI_n_jets"] >= jet_num]
        filtered_weights = data_vis["weights"][data_vis["data"]["PRI_n_jets"] >= jet_num]
        _ = data_vis["labels"][data_vis["data"]["PRI_n_jets"] >= jet_num]  # Unused in this branch

        # Drop columns containing 'PRI_n_jets' and those with zero variance
        cols_to_drop = [col for col in filtered_data.columns if "PRI_n_jets" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if np.std(filtered_data[col]) == 0]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        feature_names = list(filtered_data.columns)

    elif jet_num == 1:
        # Filter rows with exactly 1 jet
        filtered_data = data_vis["data"][data_vis["data"]["PRI_n_jets"] == jet_num]
        filtered_det_labels = data_vis["detailed_labels"][data_vis["data"]["PRI_n_jets"] == jet_num]
        print(filtered_det_labels.shape)
        _ = data_vis["labels"][data_vis["data"]["PRI_n_jets"] == jet_num]  # Unused variable
        filtered_weights = data_vis["weights"][data_vis["data"]["PRI_n_jets"] == jet_num]

        # Drop columns with 'PRI_n_jets', 'subleading', or zero variance
        cols_to_drop = [col for col in filtered_data.columns if "PRI_n_jets" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if "subleading" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if np.std(filtered_data[col]) == 0]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        feature_names = list(filtered_data.columns)

    elif jet_num == 0:
        # Filter rows with exactly 0 jets
        filtered_data = data_vis["data"][data_vis["data"]["PRI_n_jets"] == jet_num]
        filtered_det_labels = data_vis["detailed_labels"][data_vis["data"]["PRI_n_jets"] == jet_num]
        _ = data_vis["labels"][data_vis["data"]["PRI_n_jets"] == jet_num]  # Unused variable
        filtered_weights = data_vis["weights"][data_vis["data"]["PRI_n_jets"] == jet_num]

        # Drop columns with 'PRI_n_jets', 'jet', 'subleading', or zero variance
        cols_to_drop = [col for col in filtered_data.columns if "PRI_n_jets" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if "jet" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if "subleading" in col]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        cols_to_drop = [col for col in filtered_data.columns if np.std(filtered_data[col]) == 0]
        filtered_data = filtered_data.drop(columns=cols_to_drop)
        feature_names = list(filtered_data.columns)

    return filtered_data, filtered_det_labels, filtered_weights, feature_names

def createJetData(jet_num, useTestData, set_mu=3, seed=0, n_param=[1, 1, 1, 1, 1, 0], useRand=False,root_dir="/Users/ibrahim/HEP-Challenge/"):
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
    input_dir = os.path.join(root_dir, "input_data")
    program_dir = os.path.join(root_dir, "ingestion_program")
    score_dir = os.path.join(root_dir, "scoring_program")
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

def createMultiJetMultiNuanData(jet_num, useTestData, set_mu=3, seed=0, n_param=[1, 1, 1, 1, 1, 0],root_dir="/Users/ibrahim/HEP-Challenge/"):
    """
    Create multi-jet multi-nuisance data by processing multiple sub-datasets.

    Parameters:
        jet_num (int): The jet number to filter.
        useTestData (bool): Whether to use test data.
        set_mu (int, optional): Mu parameter for bootstrapping. Defaults to 3.
        seed (int, optional): Random seed. Defaults to 0.
        n_param (list, optional): List of systematic parameters. Defaults to [1,1,1,1,1,0].

    Returns:
        tuple: Processed data tensor, label tensor, weights, and feature names.
    """
    input_dir = os.path.join(root_dir, "input_data")
    program_dir = os.path.join(root_dir, "ingestion_program")
    score_dir = os.path.join(root_dir, "scoring_program")
    # Append directories so that modules from these paths can be imported
    import sys
    sys.path.append(program_dir)
    sys.path.append(score_dir)

    use_public_dataset = True
    # Import the required functions from the ingestion program
    from systematics import systematics
    from datasets import Data  # Data class for non-public dataset

    if use_public_dataset:
        from datasets import Neurips2024_public_dataset as public_dataset
        data = public_dataset()
    else:
        data = Data(input_dir)

    # Load train and test sets
    data.load_train_set()
    data.load_test_set()

    from systematics import get_bootstrapped_dataset, get_systematics_dataset

    random_state = np.random.RandomState(seed)
    test_set = data.get_test_set()

    # Create a pseudo-experimental dataset using bootstrapping
    pseudo_exp_data = get_bootstrapped_dataset(
        test_set,
        mu=set_mu,
        ttbar_scale=n_param[0],
        diboson_scale=n_param[1],
        bkg_scale=n_param[2],
        seed=seed,
        get_ans=True
    )

    weights = np.ones(pseudo_exp_data.shape[0])
    detailed_labels = pseudo_exp_data["Label"]
    pseudo_exp_data.drop(columns="Label", inplace=True)
    labels = detailed_labels[detailed_labels == "htautau"]

    print("det lab")
    print(detailed_labels)

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

    filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(jet_num, data_vis)
    temp_labels = (filtered_det_labels.values == "htautau")
    temp_labels = torch.tensor([int(val) for val in temp_labels])

    if not useTestData:
        # Compute background ratios relative to non-signal events
        ratio_ztt = len(filtered_data[filtered_det_labels == "ztautau"]) / len(filtered_data[filtered_det_labels != "htautau"])
        ratio_ttbar = len(filtered_data[filtered_det_labels == "ttbar"]) / len(filtered_data[filtered_det_labels != "htautau"])
        ratio_diboson = len(filtered_data[filtered_det_labels == "diboson"]) / len(filtered_data[filtered_det_labels != "htautau"])

        data_vis_train = data.get_train_set()
        sub_dataset = []
        sub_labels = []
        MAX_SUB_EVENTS = 10000  # Subset size per iteration

        # Process 499 sub-datasets
        for i in range(499):
            print(f"Sub-Dataset {i}")
            # Create a copy for the sub-dataset
            data_vis_sub = data_vis_train.copy()
            for key in data_vis_train.keys():
                if key != "settings":
                    try:
                        temp_df = data_vis_sub[key]
                        temp_df = temp_df.iloc[MAX_SUB_EVENTS * i: MAX_SUB_EVENTS * (i + 1)].reset_index(drop=True)
                        data_vis_sub[key] = temp_df
                    except:
                        data_vis_sub[key] = data_vis_sub[key][MAX_SUB_EVENTS * i: MAX_SUB_EVENTS * (i + 1)]

            # Apply random systematic shifts for this subset
            tes_val = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            jes_val = np.clip(random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
            soft_met_val = np.clip(random_state.lognormal(mean=0.0, sigma=1.0), a_min=0.0, a_max=5.0)

            print(tes_val, jes_val, soft_met_val)
            print(data_vis_sub["data"].shape)

            data_vis_sub_sys = systematics(
                data_set=data_vis_sub,
                tes=tes_val,
                jes=jes_val,
                soft_met=soft_met_val,
                dopostprocess=False
            )

            filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(jet_num, data_vis_sub_sys)

            count_ztt = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ztt)
            count_ttbar = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_ttbar)
            count_diboson = int(len(filtered_data[filtered_det_labels != "htautau"]) * ratio_diboson)

            temp_labels = []
            signal_data = filtered_data[filtered_det_labels == "htautau"]
            temp_labels.extend([1] * len(signal_data))
            ztt_data = filtered_data[filtered_det_labels == "ztautau"][:count_ztt]
            temp_labels.extend([0] * len(ztt_data))
            ttbar_data = filtered_data[filtered_det_labels == "ttbar"][:count_ttbar]
            temp_labels.extend([0] * len(ttbar_data))
            diboson_data = filtered_data[filtered_det_labels == "diboson"][:count_diboson]
            temp_labels.extend([0] * len(diboson_data))

            filtered_data = pd.concat((signal_data, ztt_data, ttbar_data, diboson_data), ignore_index=True)
            filtered_data = torch.tensor(filtered_data.values)
            filtered_det_labels = torch.tensor(temp_labels)

            mask = torch.any(filtered_data == -25, dim=1)
            filtered_data = filtered_data[~mask]
            filtered_det_labels = filtered_det_labels[~mask]

            # Determine columns for logarithm transform based on jet type
            if jet_num == 1:
                log_columns = [0, 3, 6, 9, 10, 13, 14, 16, 17]
            elif jet_num == 0:
                log_columns = [0, 3, 6, 9, 10, 12, 13]
            else:
                log_columns = [0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23]

            print(log_columns)
            for col_idx in range(filtered_data.shape[1]):
                if col_idx in log_columns:
                    filtered_data[:, col_idx] = torch.log(filtered_data[:, col_idx])

            sub_dataset.append(filtered_data)
            sub_labels.append(filtered_det_labels)

        # Concatenate all sub-datasets
        filtered_data = torch.cat(sub_dataset)
        filtered_det_labels = torch.cat(sub_labels)

    return filtered_data, filtered_det_labels, filtered_weights, feature_names

from torch.utils.data import Dataset, DataLoader

class Dataset1j2j(Dataset):
    """
    Custom Dataset to hold paired 1-jet and 2-jet data samples.

    Each sample is a dictionary containing:
        - 'x_2j': Data for 2-jet events.
        - 'x_1j': Data for 1-jet events.
        - 'l_2j': Labels for 2-jet events.
        - 'l_1j': Labels for 1-jet events.
    """
    def __init__(self, data_sys_list_2j, data_sys_list_1j, label_list_2j, label_list_1j):
        self.samples = []
        for i in range(len(data_sys_list_2j)):
            self.samples.append({
                'x_2j': data_sys_list_2j[i],
                'x_1j': data_sys_list_1j[i],
                'l_2j': label_list_2j[i],
                'l_1j': label_list_1j[i],
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def return1j2j(alljet_data, models,cut=False,nevents=10):
    """
    Process the input data for 1-jet and 2-jet events, apply feature transforms,
    and append normalizing flow (NF) features computed from the given models.

    Parameters:
        alljet_data (dict): Dictionary containing the combined jet data.
        models (list): List of pre-trained models for feature extraction.

    Returns:
        tuple: Data tensors and label tensors for 2-jet and 1-jet events.
    """
    # Process 2-jet events
    filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(2, alljet_data)
    temp_labels = (filtered_det_labels.values == "htautau")
    temp_labels = torch.tensor([int(val) for val in temp_labels])
    data_2j = torch.tensor(filtered_data.values)
    label_2j = torch.tensor(temp_labels)

    mask = torch.any(data_2j == -25, dim=1)
    data_2j = data_2j[~mask]
    label_2j = label_2j[~mask]

    # Log-transform specified columns for 2-jet events
    log_indices_2j = [0, 3, 6, 9, 12, 13, 24, 17, 19, 22, 23]
    for col_idx in range(data_2j.shape[1]):
        if col_idx in log_indices_2j:
            data_2j[:, col_idx] = torch.log(data_2j[:, col_idx])

    # Process 1-jet events
    filtered_data, filtered_det_labels, filtered_weights, feature_names = filterbyjet(1, alljet_data)
    temp_labels = (filtered_det_labels.values == "htautau")
    temp_labels = torch.tensor([int(val) for val in temp_labels])
    data_1j = torch.tensor(filtered_data.values)
    label_1j = torch.tensor(temp_labels)

    mask = torch.any(data_1j == -25, dim=1)
    data_1j = data_1j[~mask]
    label_1j = label_1j[~mask]

    # Log-transform specified columns for 1-jet events
    log_indices_1j = [0, 3, 6, 9, 10, 13, 14, 16, 17]
    for col_idx in range(data_1j.shape[1]):
        if col_idx in log_indices_1j:
            data_1j[:, col_idx] = torch.log(data_1j[:, col_idx])

    if cut:
        data_1j = data_1j[:nevents]
        data_2j = data_2j[:nevents]
        label_2j = label_2j[:nevents]
        label_1j = label_1j[:nevents]

    # Compute NF features from the provided models
    with torch.no_grad():
        NF_feat_s1j = torch.sigmoid(models[3](data_1j)).cpu().unsqueeze(1)
        NF_feat_b1j = torch.sigmoid(models[0](data_1j)).cpu().unsqueeze(1)
        NF_feat_s1j_3 = torch.sigmoid(models[2](data_1j)).cpu().unsqueeze(1)
        NF_feat_b1j_3 = torch.sigmoid(models[1](data_1j)).cpu().unsqueeze(1)

        NF_feat_s2j = torch.sigmoid(models[7](data_2j)).cpu().unsqueeze(1)
        NF_feat_b2j = torch.sigmoid(models[4](data_2j)).cpu().unsqueeze(1)
        NF_feat_s2j_3 = torch.sigmoid(models[6](data_2j)).cpu().unsqueeze(1)
        NF_feat_b2j_3 = torch.sigmoid(models[5](data_2j)).cpu().unsqueeze(1)

        # Append the NF features to the original data
        data_2j = torch.cat([data_2j, NF_feat_s2j, NF_feat_s2j_3, NF_feat_b2j, NF_feat_b2j_3], dim=1)
        data_1j = torch.cat([data_1j, NF_feat_s1j, NF_feat_s1j_3, NF_feat_b1j, NF_feat_b1j_3], dim=1)

    return data_2j, data_1j, label_2j, label_1j

def load_nf_models(models_dir,device):
    """
    Load NormalizingFlowModel models from a directory structure.

    The expected structure is:
        models_dir/
            1_jet/
                *.ckpt   # models for 1 jet (indices 0-3)
            2_jet/
                *.ckpt   # models for 2 jets (indices 4-7)

    Returns:
        A list of loaded models in order (first the 1_jet models, then the 2_jet models).
    """
    models = []
    ckpt_path_names = []
    # Load models for 1 jet
    one_jet_dir = os.path.join(models_dir, "1_jet")
    if not os.path.isdir(one_jet_dir):
        raise FileNotFoundError(f"Directory not found: {one_jet_dir}")
    # Sorting ensures consistent order if file names are compatible.
    for ckpt_file in sorted(os.listdir(one_jet_dir)):
        if ckpt_file.endswith(".ckpt"):
            ckpt_path = os.path.join(one_jet_dir, ckpt_file)
            model = NormalizingFlowModel.load_from_checkpoint(ckpt_path).to(device).eval().to(torch.float32)
            models.append(model)
            ckpt_path_names.append(ckpt_path)
    
    # Load models for 2 jets
    two_jet_dir = os.path.join(models_dir, "2_jet")
    if not os.path.isdir(two_jet_dir):
        raise FileNotFoundError(f"Directory not found: {two_jet_dir}")
    for ckpt_file in sorted(os.listdir(two_jet_dir)):
        if ckpt_file.endswith(".ckpt"):
            ckpt_path = os.path.join(two_jet_dir, ckpt_file)
            model = NormalizingFlowModel.load_from_checkpoint(ckpt_path).to(device).eval().to(torch.float32)
            models.append(model)
            ckpt_path_names.append(ckpt_path)
    
    print("Loaded models from: ")
    print(ckpt_path_names)
    return models
