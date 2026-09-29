import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
FILE_PATH = 'drainage_data.csv'
OUTPUT_DIR = 'EDA_Plots'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading dataset from {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)

# Data Cleaning: 
# Remove early initialization glitches (e.g., elapsed_ms < 4000)
df = df.drop(df[df['elapsed_ms'] < 4000].index)
df = df.sort_values('elapsed_ms')

# Convert milliseconds to seconds for plotting
df['time_sec'] = df['elapsed_ms'] / 1000.0

# Define color scheme for the four statuses
colors = {
    'NORMAL': '#2ca02c',   # Green
    'WARNING': '#ff7f0e',  # Orange
    'HVYRAIN': '#1f77b4',  # Blue
    'OBSTRUCT': '#d62728'  # Red
}

print("Generating Class Distribution Bar Chart...")
# 1. Bar Chart: Drainage Status Distribution
plt.figure(figsize=(8, 5))
counts = df['status'].value_counts()
counts.plot(kind='bar', color=[colors.get(x, 'gray') for x in counts.index])
plt.title('Drainage Status Distribution')
plt.ylabel('Number of Records')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'status_distribution.png'))
plt.close()

print("Generating Feature Space Scatter Plot...")
# 2. Scatter Plot: Distance vs. Flow Rate
plt.figure(figsize=(9, 6))
for status, color in colors.items():
    subset = df[df['status'] == status]
    plt.scatter(subset['flow_l_min'], subset['distance_cm'], label=status, color=color, alpha=0.6, edgecolors='w')

# Draw Threshold Lines
plt.axhline(y=23, color='k', linestyle='--', alpha=0.5, label='Warning Threshold (23cm)')
plt.axhline(y=21, color='r', linestyle='--', alpha=0.5, label='Critical Threshold (21cm)')
plt.axvline(x=3.5, color='b', linestyle='--', alpha=0.5, label='Flow Threshold (3.5 L/min)')

# Invert Y-axis because a lower distance means a higher water level
plt.gca().invert_yaxis()
plt.title('Sensor Feature Space: Distance vs Flow Rate')
plt.xlabel('Flow Rate (L/min)')
plt.ylabel('Ultrasonic Distance (cm) [Inverted: Higher = Lower Water Level]')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distance_vs_flow.png'))
plt.close()

print("Generating Time Series Plot...")
# 3. Time Series Plot: Simulated Drainage Event
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(df['time_sec'], df['distance_cm'], color='purple', label='Distance (cm)')
ax2.plot(df['time_sec'], df['flow_l_min'], color='teal', label='Flow (L/min)')

ax1.set_xlabel('Elapsed Time (seconds)')
ax1.set_ylabel('Distance (cm)', color='purple')
ax2.set_ylabel('Flow Rate (L/min)', color='teal')

# Invert the distance axis so visually, the line goes UP when water level goes UP
ax1.invert_yaxis()
plt.title('Simulated Drainage Event Over Time')
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'time_series.png'))
plt.close()

print(f"All EDA plots successfully generated and saved to the '{OUTPUT_DIR}' folder!")
