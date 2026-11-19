
import re
import matplotlib.pyplot as plt

def parse(input_file):
    samples = []
    accuracy = []

    with open(input_file, "r") as file:
        for line in file:
            match = re.search(
                r"iter:(\d+)/\d+,\s*tta_acc1=([\d.]+)",
                line
            )

            if match:
                samples.append(int(match.group(1)))
                accuracy.append(float(match.group(2)))

    # Remove duplicate sample counts, keeping the last value
    data = dict(zip(samples, accuracy))

    return list(data.keys()), list(data.values())


# Add all your log files here
log_files = {
    "USE + reward feedback": "results1.txt",
    "USE": "results2.txt",
    # "Method 4": "results4.txt",
}

# Plot all methods on the same graph
plt.figure(figsize=(12, 6))

for method, input_file in log_files.items():
    samples, accuracy = parse(input_file)

    plt.plot(
        samples,
        accuracy,
        marker="o",
        linestyle="-",
        markersize=3,
        linewidth=1.5,
        label=method
    )

    # Display values at each point
    for x, y in zip(samples, accuracy):
        plt.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=6
        )

# Labels and title
plt.xlabel("Number of Test Samples", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Accuracy of USE vs USE + reward feedback on DTD dataset", fontsize=14)

# Legend and grid
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save and display
plt.savefig("accuracy_comparison2.png", dpi=300, bbox_inches="tight")
plt.show()

print("Graph saved as accuracy_comparison.png")
