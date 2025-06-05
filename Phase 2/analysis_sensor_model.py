import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict

# Path to your folder
folder_path = "proximity"

# Organize files by distance
distance_groups = defaultdict(list)

# Regex pattern to extract distance and condition
pattern = re.compile(r"measures_(\w+)_(\w+)\.csv")

# Group files
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = pattern.match(filename)
        if match:
            distance, condition = match.groups()
            distance_groups[distance].append((condition, filename))

# Plot per distance group
for distance, files in distance_groups.items():
    plt.figure(figsize=(10, 6))

    for condition, filename in files:
        filepath = os.path.join(folder_path, filename)
        data = pd.read_csv(filepath, header=None)[0]

        # Plot histogram
        sns.histplot(data, bins=50, kde=False, stat="density",
                     label=f"{condition} (hist)", alpha=0.3)

        # Plot KDE
        sns.kdeplot(data, linewidth=2, label=f"{condition} (kde)", alpha=0.7)

    plt.title(f"Histogram and PDF for distance: {distance}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(folder_path, f"{distance}_comparison.png")
    plt.savefig(output_file)
    plt.close()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict

# Path to your folder
folder_path = "proximity"

# Organize files by distance
distance_groups = defaultdict(list)

# Regex to extract distance and condition from filenames like measures_near_control.csv
pattern = re.compile(r"measures_(\w+)_(\w+)\.csv")

# Group files
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = pattern.match(filename)
        if match:
            distance, condition = match.groups()
            distance_groups[distance].append((condition, filename))

# Plot per distance group
for distance, files in distance_groups.items():
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Histograms and PDFs for distance: {distance}", fontsize=16)

    for idx, (condition, filename) in enumerate(files):
        filepath = os.path.join(folder_path, filename)
        data = pd.read_csv(filepath, header=None)[0]

        ax = plt.subplot(2, 2, idx + 1)
        sns.histplot(data, bins=50, kde=False, stat="density",
                     color="skyblue", alpha=0.5, ax=ax, label="Histogram")
        sns.kdeplot(data, color="red", linewidth=2, ax=ax, label="KDE")

        ax.set_title(f"Condition: {condition}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    # Save figure
    output_file = os.path.join(folder_path, f"{distance}_2x2.png")
    plt.savefig(output_file)
    plt.close()
# Compute and save summary table: mean ± std (with overall, case-insensitive)
summary_stats = defaultdict(dict)

for distance, files in distance_groups.items():
    all_values = []  # collect all values across conditions for this distance
    for condition, filename in files:
        condition = condition.lower()  # Normalize condition name
        filepath = os.path.join(folder_path, filename)
        data = pd.read_csv(filepath, header=None)[0]

        all_values.append(data)

        mean_val = data.mean()
        std_val = data.std()
        summary_stats[distance][condition] = f"{mean_val:.2f} ± {std_val:.2f}"

    # Compute overall mean and std
    combined_data = pd.concat(all_values, ignore_index=True)
    overall_mean = combined_data.mean()
    overall_std = combined_data.std()
    summary_stats[distance]["overall"] = f"{overall_mean:.2f} ± {overall_std:.2f}"

# Convert to DataFrame
summary_df = pd.DataFrame.from_dict(summary_stats, orient="index")
summary_df.index.name = "Distance"

# Save to CSV
summary_df.to_csv(os.path.join(folder_path, "summary_mean_std.csv"))

# Print to console
print("Summary of Mean ± Std (including overall):")
print(summary_df)
