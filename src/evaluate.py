from sklearn.metrics import average_precision_score, roc_auc_score
from scipy import stats
import numpy as np

def evaluate_ap(y_true, y_score):
    return average_precision_score(y_true, y_score)

def evaluate_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)


def summarize_results(df):
    return df.groupby(['mode', 'model']).agg(
        val_AP_mean=('val_AP', 'mean'),
        val_AP_std=('val_AP', 'std'),
        val_AUC_mean=('val_AUC', 'mean'),
        val_AUC_std=('val_AUC', 'std'),
        test_AP_mean=('test_AP', 'mean'),
        test_AP_std=('test_AP', 'std'),
        test_AUC_mean=('test_AUC', 'mean'),
        test_AUC_std=('test_AUC', 'std'),
        time_mean=('time_sec', 'mean'),
    ).reset_index()

def paired_ttest(df, model_a, model_b, mode, metric='test_AP'):
    a = df[(df['mode']==mode)&(df['model']==model_a)].sort_values('seed')[metric].values
    b = df[(df['mode']==mode)&(df['model']==model_b)].sort_values('seed')[metric].values
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    return {'mean_diff': diff.mean(), 'std_diff': diff.std(ddof=1),
            't': t, 'p': p, 'n': len(a)}