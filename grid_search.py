import yaml
import subprocess
import os
import itertools

video_thresholds = [0.1, 0.2, 0.4]
audio_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
frame_skips = [15, 30]
ensemble_thresholds = [0.3, 0.4, 0.5, 0.6] 
ensemble_methods = ["mean", "weighted_average", "majority_voting", "weighted_voting", "stacking"] 

original_yaml_path = "./backend/config/ensemble.yaml"
main_script_name = "backend/main.py" 
base_output_dir = "./experiment_results"

def run_grid_search_sequential_folders():
    with open(original_yaml_path, "r") as file:
        base_config = yaml.safe_load(file)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    print("Starting Sequential Grid Search...")

    # Generate all possible combinations cleanly
    combinations = list(itertools.product(
        video_thresholds, 
        audio_thresholds, 
        frame_skips,
        ensemble_thresholds,
        ensemble_methods
    ))
    
    total_runs = len(combinations)
    print(f"Total combinations to test: {total_runs}")

    for idx, (v_thresh, a_thresh, f_skip, e_thresh, method) in enumerate(combinations, 1):
        print(f"Run {idx}/{total_runs} | V: {v_thresh} | A: {a_thresh} | FS: {f_skip} | E_Thresh: {e_thresh} | Method: {method}")
        
        # Create unique folder including all parameters
        folder_name = f"v{v_thresh}_a{a_thresh}_fs{f_skip}_e{e_thresh}_{method}"
        folder_path = os.path.join(base_output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        temp_yaml_path = os.path.join(folder_path, "config.yaml")
        output_file = os.path.join(folder_path, "summary.txt")

        # Update and save config
        config = base_config.copy()
        config["video_decision_threshold"] = v_thresh
        config["audio_decision_threshold"] = a_thresh
        config["frame_skip"] = f_skip
        config["ensemble_confidence_decision_threshold"] = e_thresh
        config["ensemble_method"] = method
        
        with open(temp_yaml_path, "w") as file:
            yaml.dump(config, file)

        # Run sequentially
        with open(output_file, "w") as f:
            f.write(f"Testing V: {v_thresh} | A: {a_thresh} | FS: {f_skip} | E_Thresh: {e_thresh} | Method: {method}\n{'='*50}\n")
            subprocess.run(
                ["python", main_script_name, "--config", temp_yaml_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
        
        print(f"Finished. Saved to {folder_path}\n")

if __name__ == "__main__":
    run_grid_search_sequential_folders()