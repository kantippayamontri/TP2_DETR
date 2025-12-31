import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def draw_absolute():
    # Load the annotation files
    activitynet_path = Path("../../../GAP/data/ActivityNet13/ActivityNet13_annotations.json")
    thumos14_path = Path("../../../GAP/data/Thumos14/Thumos14_annotations.json")

    with open(activitynet_path, "r") as f:
        activitynet_data = json.load(f)

    with open(thumos14_path, "r") as f:
        thumos14_data = json.load(f)

    # Extract action durations from ActivityNet
    activitynet_durations = []
    for video in activitynet_data["database"].values():
        if "annotations" in video:
            for ann in video["annotations"]:
                start, end = ann["segment"]
                activitynet_durations.append(end - start)

    # Extract action durations from Thumos14
    thumos14_durations = []
    for video in thumos14_data["database"].values():
        if "annotations" in video:
            for ann in video["annotations"]:
                start, end = ann["segment"]
                thumos14_durations.append(end - start)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(activitynet_durations, bins=30, alpha=0.7, label="ActivityNet1.3", color="skyblue", edgecolor="black")
    plt.hist(thumos14_durations, bins=30, alpha=0.7, label="Thumos14", color="salmon", edgecolor="black")
    plt.axvline(x=10, color='gray', linestyle='--', label='10s threshold (short actions)')
    plt.xlabel("Action Duration (seconds)")
    plt.ylabel("Number of Actions")
    plt.title("Comparison of Action Duration Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('statistics(absolute-pure_count).png')

    # Define duration threshold (in seconds)
    SHORT_THRESHOLD = 10.0
    LONG_THRESHOLD = 60.0  # Optional: anything >= 60s could be considered long

    # Categorize durations
    def categorize_durations(durations):
        short = [d for d in durations if d < SHORT_THRESHOLD]
        medium = [d for d in durations if SHORT_THRESHOLD <= d < LONG_THRESHOLD]
        long = [d for d in durations if d >= LONG_THRESHOLD]
        return short, medium, long

    an_short, an_med, an_long = categorize_durations(activitynet_durations)
    th_short, th_med, th_long = categorize_durations(thumos14_durations)

    # Bar chart comparison (absolute count)
    labels = ['Short (<10s)', 'Medium (10-60s)', 'Long (>=60s)']
    an_counts = [len(an_short), len(an_med), len(an_long)]
    th_counts = [len(th_short), len(th_med), len(th_long)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, an_counts, width, label='ActivityNet1.3', color='skyblue')
    rects2 = ax.bar(x + width/2, th_counts, width, label='Thumos14', color='salmon')

    ax.set_ylabel('Number of Actions')
    ax.set_title('Action Duration Category Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('statistics(absolute-scale_count).png')

    # Relative proportion chart
    an_total = sum(an_counts)
    th_total = sum(th_counts)

    an_props = [c / an_total for c in an_counts]
    th_props = [c / th_total for c in th_counts]

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, an_props, width, label='ActivityNet1.3', color='skyblue')
    rects2 = ax.bar(x + width/2, th_props, width, label='Thumos14', color='salmon')

    ax.set_ylabel('Proportion of Actions')
    ax.set_title('Relative Distribution of Action Durations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('statistics(absolute-scale_proportion).png')


def draw_relative():
    # Load the annotation files
    activitynet_path = Path("../../../GAP/data/ActivityNet13/ActivityNet13_annotations.json")
    thumos14_path = Path("../../../GAP/data/Thumos14/Thumos14_annotations.json")

    with open(activitynet_path, "r") as f:
        activitynet_data = json.load(f)

    with open(thumos14_path, "r") as f:
        thumos14_data = json.load(f)

    # 計算 relative duration
    def get_relative_durations(data):
        relative_durations = []
        for video in data["database"].values():
            duration = video["duration"]
            if "annotations" in video and duration > 0:
                for ann in video["annotations"]:
                    start, end = ann["segment"]
                    relative = (end - start) / duration
                    relative_durations.append(relative)
        return relative_durations

    an_rel = get_relative_durations(activitynet_data)
    th_rel = get_relative_durations(thumos14_data)

    # 分 bin 統計 (每 5% 一格)
    bins = [i / 100 for i in range(0, 105, 5)]
    bin_labels = [f"{i}–{i+5}%" for i in range(0, 100, 5)]
    x = np.arange(len(bin_labels))

    an_hist, _ = np.histogram(an_rel, bins=bins)
    th_hist, _ = np.histogram(th_rel, bins=bins)

    # 繪圖
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, an_hist, width=width, label="ActivityNet1.3", color="skyblue")
    plt.bar(x + width/2, th_hist, width=width, label="Thumos14", color="salmon")
    plt.xticks(x, bin_labels, rotation=45)
    plt.xlabel("Relative Duration of Action (% of Video)")
    plt.ylabel("Number of Actions")
    plt.title("Relative Duration Distribution (Binned by 5%)")
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('statistics(relative).png')



if __name__=='__main__':

    draw_absolute()
    draw_relative()

