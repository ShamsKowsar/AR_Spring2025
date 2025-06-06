import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Translation.csv')

dx = [50, 100, 150]   # mm
v = 30    # mm/s
dt = [x/v for x in dx]     # s

df['speed of 50 mm'] = df['50 mm'].apply(lambda x: x/dt[0])
df['speed of 100 mm'] = df['100 mm'].apply(lambda x: x/dt[1])
df['speed of 150 mm'] = df['150 mm'].apply(lambda x: x/dt[2])

print(f'mean value of dx for 50 mm: {df["50 mm"].mean():.2f}')
print(f'mean value of dx for 100 mm: {df["100 mm"].mean():.2f}')
print(f'mean value of dx for 150 mm: {df["150 mm"].mean():.2f}')

print(f'\n\nvariance value of dx for 50 mm: {df["50 mm"].var():.2f}')
print(f'variance value of dx for 100 mm: {df["100 mm"].var():.2f}')
print(f'variance value of dx for 150 mm: {df["150 mm"].var():.2f}')


print(f'mean value of speed for 50 mm: {df["speed of 50 mm"].mean():.2f}')
print(f'mean value of speed for 100 mm: {df["speed of 100 mm"].mean():.2f}')
print(f'mean value of speed for 150 mm: {df["speed of 150 mm"].mean():.2f}')

print(f'\n\nvariance value of speed for 50 mm: {df["speed of 50 mm"].var():.2f}')
print(f'variance value of speed for 100 mm: {df["speed of 100 mm"].var():.2f}')
print(f'variance value of speed for 150 mm: {df["speed of 150 mm"].var():.2f}')


# plotting histogram and distribution of speeds

for x, v in zip(dx, df.columns[3:]):
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  sns.histplot(df[v], kde=False, bins=15, color='skyblue')
  plt.title(f"Histogram of Speed (Distance {x} mm)")
  plt.xlabel("Speed (mm/s)")
  plt.ylabel("Frequency")

  plt.subplot(1, 2, 2)
  sns.kdeplot(df[v], color='red', fill=True)
  plt.title(f"PDF of Speed (Distance {x} mm)")
  plt.xlabel("Speed (mm/s)")
  plt.ylabel("Density")

  plt.tight_layout()
  plt.show()

