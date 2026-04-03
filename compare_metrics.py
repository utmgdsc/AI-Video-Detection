import os
import re
import csv

# Ensure this matches the folder containing your v0.1_... folders
base_dir = "./result/balanced_test_set_experiment_results" 
output_csv = "comprehensive_metrics_comparison.csv"

models = ["EfficientNet", "MesoNet", "XceptionNet", "AASIST", "Ensemble"]

# Regex for Accuracy
acc_patterns = {
    "EfficientNet": re.compile(r"EfficientNet accuracy:\s*([0-9.]+)"),
    "MesoNet": re.compile(r"MesoNet accuracy:\s*([0-9.]+)"),
    "XceptionNet": re.compile(r"XeceptionNet accuracy:\s*([0-9.]+)"), # Matches typo
    "AASIST": re.compile(r"AAsist accuracy:\s*([0-9.]+)"),
    "Ensemble": re.compile(r"Ensemble accuracy:\s*([0-9.]+)")
}

# Regex for Metrics (Precision, Recall, TN, FP, FN, TP) using DOTALL to read across lines
metrics_patterns = {}
for m in models:
    # Captures: 1=Precision, 2=Recall, 3=TN, 4=FP, 5=FN, 6=TP
    pattern = re.compile(
        rf"{m} metrics:.*?Precision:\s*([0-9.]+).*?Recall:\s*([0-9.]+).*?Actual Real\s+(\d+)\s+(\d+).*?Actual Fake\s+(\d+)\s+(\d+)", 
        re.DOTALL | re.IGNORECASE
    )
    metrics_patterns[m] = pattern

results = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Extract hyperparameters from the folder name
    try:
        parts = folder_name.split('_')
        v = parts[0].replace('v', '')
        a = parts[1].replace('a', '')
        fs = parts[2].replace('fs', '')
    except Exception:
        v, a, fs = "N/A", "N/A", "N/A"

    summary_file = os.path.join(folder_path, "summary.txt")
    if not os.path.exists(summary_file):
        continue

    # Initialize row with N/A
    row_data = {
        "Folder": folder_name,
        "Video_Thresh": v,
        "Audio_Thresh": a,
        "Frame_Skip": fs
    }
    
    for m in models:
        row_data[f"{m}_Acc"] = "N/A"
        row_data[f"{m}_Prec"] = "N/A"
        row_data[f"{m}_Rec"] = "N/A"
        row_data[f"{m}_TN"] = "N/A"
        row_data[f"{m}_FP"] = "N/A"
        row_data[f"{m}_FN"] = "N/A"
        row_data[f"{m}_TP"] = "N/A"

    # Search the text file
    with open(summary_file, "r") as f:
        content = f.read()
        
        # 1. Grab Accuracies
        for m, pattern in acc_patterns.items():
            match = pattern.search(content)
            if match:
                row_data[f"{m}_Acc"] = match.group(1)

        # 2. Grab Metrics and Confusion Matrix
        for m, pattern in metrics_patterns.items():
            match = pattern.search(content)
            if match:
                row_data[f"{m}_Prec"] = match.group(1)
                row_data[f"{m}_Rec"] = match.group(2)
                row_data[f"{m}_TN"] = match.group(3)
                row_data[f"{m}_FP"] = match.group(4)
                row_data[f"{m}_FN"] = match.group(5)
                row_data[f"{m}_TP"] = match.group(6)

    results.append(row_data)

# Sort results so the highest Ensemble accuracy is at the top
results.sort(key=lambda x: float(x['Ensemble_Acc']) if x['Ensemble_Acc'] != 'N/A' else -1.0, reverse=True)

# Dynamically build CSV headers
headers = ["Folder", "Video_Thresh", "Audio_Thresh", "Frame_Skip"]
for m in models:
    headers.extend([f"{m}_Acc", f"{m}_Prec", f"{m}_Rec", f"{m}_TN", f"{m}_FP", f"{m}_FN", f"{m}_TP"])

# Export to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(results)

print(f"Extraction complete! Extensive metrics saved to {output_csv}")