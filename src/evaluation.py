import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

# Plot ROC curve
def plot_roc_curve(model, X_test, y_test, save_path=None):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC: {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Check bias based on Gender
def check_bias(data, model, X_test, y_test, save_path=None):
    if 'Gender' in data.columns:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        bias_data = data.loc[X_test.index]
        bias_data['Predicted Probability'] = y_pred_proba

        plt.figure()
        sns.barplot(x='Gender', y='Predicted Probability', data=bias_data)
        plt.title('Predicted Probability of Mental Illness by Gender')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
