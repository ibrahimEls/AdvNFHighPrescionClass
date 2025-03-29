import os
import sys
import json
import argparse
import numpy as np
import torch

# Import functions from your custom libraries.
from lib.data_utils import return1j2j, load_nf_models
from lib.stat_utils import compute_mu_nuan_2NP_class, fit_2D_splines_bin_by_bin_from_dict, string_to_tuple_str,load_bias_data,get_confidence_interval
from lib.class_model import CombinedClassifier  # Assumed classifier model to load


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

def main(args, data):
    
    print(f"Running pipline on test dataset with {args.mu}")
    # --------------------------
    # Step 1: Load histogram templates.
    # --------------------------
    with open(args.hist_path, 'r') as f:
        serializable_dict = json.load(f)
    
    # Convert the loaded dictionary to one with numpy arrays.
    hist_dict = {
        key: (np.array(v['sig']), np.array(v['bg']))
        for key, v in serializable_dict.items()
    }
    
    # Create dictionaries mapping parameter tuples to signal and background arrays.
    S_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][0] for i in hist_dict.keys()}
    B_templates_2d_2j = {string_to_tuple_str(i): hist_dict[i][1] for i in hist_dict.keys()}
    
    # Fit 2D splines bin-by-bin using the dictionaries.
    bin_splines_S_class = fit_2D_splines_bin_by_bin_from_dict(S_templates_2d_2j)
    bin_splines_BG_class = fit_2D_splines_bin_by_bin_from_dict(B_templates_2d_2j)
    
    # loading Neyman data
    std_corrected_interp, a, b = load_bias_data(args.neyman_path)

    # --------------------------
    # Step 2: Load models and classifier.
    # --------------------------
    # Load Normalizing Flow models from the provided directory.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    models = load_nf_models(args.models_dir,device)
    
    # Load the classifier model from its checkpoint.
    class_model_load = CombinedClassifier.load_from_checkpoint(args.class_model_path).to(device).eval().to(torch.float32)
    

    alljet_data, _ = createJetData("all", True, set_mu=args.mu, seed=31245, n_param=[1, 1, 1, 1, 1, 0], useRand=True, data=data)
    if not args.predict_numevents:
        # Split the data into 2-jet and 1-jet subsets.
        data_2j, data_1j, label_2j, label_1j = return1j2j(alljet_data, models)
        
        # Compute the MLE mu using the provided classifier and fitted splines.
        mu = compute_mu_nuan_2NP_class(data_2j,data_1j, class_model_load, bin_splines_S_class,bin_splines_BG_class)
        mu_MLE,mu_lower, mu_upper = get_confidence_interval(mu,std_corrected_interp,a,b)

        print(f"Real Mu {args.mu}")
        print(f"Predicted Post-Process Mu {mu_MLE}")
        print(f"Predicted Post-Process CI of Mu {mu_lower}, {mu_upper}")
        print(f"Predicted Post-Process delta_mu_hat {abs(mu_upper - mu_lower)/2}")
        
    else:
        print(f"Running classifer (not a mu estimate!) for {args.nevent}")
        data_2j, data_1j, label_2j, label_1j = return1j2j(alljet_data, models,cut=True,nevents=args.nevent)

        with torch.no_grad():
            scores_2j = torch.sigmoid(class_model_load(data_2j,2)).cpu().numpy()
            scores_1j = torch.sigmoid(class_model_load(data_1j,1)).cpu().numpy()

        print("2J Predection:")
        print(scores_2j)
        print("2J True Labels:")
        print(label_2j)

        print("1J Predection:")
        print(scores_1j)
        print("1J True Labels:")
        print(label_1j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute MLE ratios for jet data using NF models and a classifier."
    )
    # Path to load the input histogram JSON file.
    parser.add_argument(
        "--hist_path",
        type=str,
        default="PreTrained/SavedStats/histogram_2jet1jet_class_4nf_200bins.json",
        help="Path to the input histogram JSON file."
    )
    # Path to save the output JSON file.
    parser.add_argument(
        "--json_save_path", 
        type=str, 
        default="Test/SavedStats/",
        help="Path to save the resulting JSON file."
    )
    # Root directory containing input data and subdirectories for ingestion/scoring.
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="/Users/ibrahim/HEP-Challenge/",
        help="Root directory path for data files."
    )
    # Path to the classifier model checkpoint.
    parser.add_argument(
        "--class_model_path", 
        type=str, 
        default="PreTrained/Models/DNN/DNNclass4NF_2.ckpt",
        help="Path to load classifier model checkpoint using CombinedClassifier.load_from_checkpoint."
    )
    # Directory where the Normalizing Flow models are stored.
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="PreTrained/Models/",
        help="Path to the directory containing the '1_jet' and '2_jet' subdirectories with checkpoint files."
    )

    parser.add_argument(
        "--neyman_path", 
        type=str, 
        default="PreTrained/SavedStats/2jet_1jet_2NP_MLE_Final_DNN_test_4NF_200.json",
        help="Path to the Neyman JSON"
    )
    
    parser.add_argument(
        "--mu", 
        type=float, 
        default=1,
        help="mu"
    )

    parser.add_argument(
        "--predict_numevents", 
        action="store_true",
        help="predict mu on a test dataset"
    )

    parser.add_argument(
        "--nevent", 
        type=int, 
        default=10,
        help="Number of events to test if predict_mu_test is False"
    )


    args = parser.parse_args()
    
    # Set up paths for additional program directories.
    input_dir = os.path.join(args.root_dir, "input_data")
    program_dir = os.path.join(args.root_dir, "ingestion_program")
    score_dir = os.path.join(args.root_dir, "scoring_program")
    
    # Append ingestion and scoring program directories to sys.path for module imports.
    sys.path.append(program_dir)
    sys.path.append(score_dir)
    
    use_public_dataset = True
    # Import dataset classes from the ingestion program.
    from systematics import systematics
    from datasets import Data  # Data class for non-public dataset
    if use_public_dataset:
        from datasets import Neurips2024_public_dataset as public_dataset
        data = public_dataset()
    else:
        data = Data(input_dir)
    
    print("Loading data ...")
    data.load_train_set()
    data.load_test_set()
    
    # Call the main processing function.
    main(args, data)