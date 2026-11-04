from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_ap(y_true, y_score):
    return average_precision_score(y_true, y_score)

def evaluate_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def summarize_results(df):
    return df.groupby(['mode', 'model']).agg(
        val_AP_mean=('val_AP', 'mean'),
        val_AP_std=('val_AP', 'std'),
        val_AUC_mean=('val_AUC', 'mean'),
        val_AUC_std=('val_AUC', 'std'),
        time_mean=('time_sec', 'mean'),
    ).reset_index()