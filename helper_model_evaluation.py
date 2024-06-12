import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve


def plot_confusion(y_true, y_pred, perc=None):
    """For binary classifications only, where y_true an array-like of 0 or 1. 
    y_pred is an array-like of float between 0 and 1. 
    perc is the threshold: if y_pred >= perc, then predict 1, otherwise 0. 

    Args:
        y_true (array-like): correct target value
        y_pred (array-like): predicted value
        perc (float, optional): The threshold. Defaults to None.
    """
    if perc is None:
        cm = confusion_matrix(y_true, y_pred)
    else:
        cl = []
        for pred in y_pred:
            if pred >= perc:
                cl.append(1)
            else:
                cl.append(0)
        print(classification_report(y_true, cl))
        print(f"Threshold Used: {perc}")
        cm = confusion_matrix(y_true, cl)
    # plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_roc(name, labels, predictions, **kwargs):
    """Plotting roc curve. 

    Args:
        name (str): Name of plot.
        labels (array-like): correct target value
        predictions (array-like): predicted value
    """
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5,20])
    # plt.ylim([-0.5,20])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
