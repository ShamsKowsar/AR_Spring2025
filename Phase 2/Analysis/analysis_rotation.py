import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    # Clean angle column (e.g., '77/5' → 15.4)
    def parse_angle(val):
        try:
            return float(val)
        except ValueError:
            return eval(str(val).replace("/", "."))
        
    df["Angle"] = df["Angle"].apply(parse_angle)
    return df

def analyze_and_plot(df):
    results = {}
    for direction in ["CW", "CCW"]:
        angles = df[df["Direction"] == direction]["Angle"]
        mean_val = angles.mean()
        var_val = angles.var(ddof=1)
        results[direction] = {
            "mean": mean_val,
            "variance": var_val,
            "data": angles
        }
        print(f"{direction} → Mean: {mean_val:.2f}, Variance: {var_val:.2f}")

    # Plot
    for direction in ["CW", "CCW"]:
        angles = results[direction]["data"]
        mu = results[direction]["mean"]
        sigma = np.sqrt(results[direction]["variance"])

        plt.figure(figsize=(8, 5))
        plt.hist(angles, bins=6, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        x = np.linspace(min(angles) - 5, max(angles) + 5, 100)
        plt.plot(x, norm.pdf(x, mu, sigma), 'r', linewidth=2)

        plt.title(f"Rotation Angle Distribution - {direction}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Probability Density")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{direction}.png')
        plt.show()

# === Run analysis ===
csv_file = "Rotation.csv"  # Replace with your file path
df = load_and_clean_data(csv_file)
analyze_and_plot(df)
