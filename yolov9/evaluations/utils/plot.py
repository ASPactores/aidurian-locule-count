# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_precision_recall_curve(metrics, title):
    precision = metrics.seg.curves_results[0][1][0]
    recall = metrics.seg.curves_results[0][0]
    area_under_curve = metrics.seg.map50

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve", color="black")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: " + title)

    # Move legend outside the plot
    plt.legend(
        [f"Locules: {area_under_curve:.3f}"], loc="upper right", bbox_to_anchor=(1.3, 1)
    )

    # Remove top and right border
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()
