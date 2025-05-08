import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# === 可视化工具 ===
def plot_loss_values(loss_values, x_label, y_label, title="Loss over Epochs"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Loss")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_roc_curve(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_pr_curve(true_labels, scores):
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 5))
    plt.plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
