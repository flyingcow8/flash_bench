import yaml
from collections import OrderedDict

def ordered_dict_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, ordered_dict_representer)

def append_results_to_yaml(
    params, fwd_results, best_fwd_solution, bwd_results, best_bwd_solution, output_file
):
    # Create a new dictionary to maintain the desired order
    ordered_params = OrderedDict()

    # Copy existing keys from params
    for key in params:
        ordered_params[key] = params[key]

    ordered_params["fwd_results"] = [
        str(list(result.values())) for result in fwd_results
    ]
    ordered_params["best_fwd_solution"] = str(list(best_fwd_solution.values()))

    if params.get("is_training", True):
        ordered_params["bwd_results"] = [
            str(list(result.values())) for result in bwd_results
        ]
        ordered_params["best_bwd_solution"] = str(list(best_bwd_solution.values()))

    with open(output_file, "a") as f:
        yaml.dump([ordered_params], f, default_flow_style=False)
        f.write("\n")  # Add a newline for better readability between entries