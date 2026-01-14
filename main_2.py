import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from scipy.optimize import minimize
from scipy.stats import rankdata, gmean
from math import pi
from itertools import cycle
import sys

#Data Loading and Standardization Functions

def load_and_standardize(filepath, model_name, is_four_class=False):
    """
    Loads a CSV, ensures columns are [Healthy, Unhealthy, Rubbish],
    normalizes probabilities, and returns a cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Standardize column names (strip whitespace, lower case check)
    df.columns = [c.strip() for c in df.columns]
    
    
    rename_map = {
        'Unhealthy_prob': 'Unhealthy',
        'healthy': 'Healthy',
        'unhealthy': 'Unhealthy',
        'rubbish': 'Rubbish',
        'label': 'predicted_label'
    }
    df.rename(columns=rename_map, inplace=True)
    
    req_cols = ['Healthy', 'Unhealthy', 'Rubbish']
    
    # Handle the 4-class case
    if is_four_class:
        if 'bothcells_prob' in df.columns:
            df = df.drop(columns=['bothcells_prob'])
        
        # Check columns exist
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"WARNING: Model {model_name} missing columns: {missing}. Skipping.")
            return None
            
        # Re-normalize probabilities
        probs = df[req_cols].values
        row_sums = probs.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        probs = probs / row_sums
        df[req_cols] = probs
    
    # Ensure columns exist for normal cases
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Model {model_name} missing columns: {missing}. Skipping.")
        return None

    if 'image_name' not in df.columns:
         print(f"WARNING: Model {model_name} missing 'image_name' column. Skipping.")
         return None

    df = df[['image_name'] + req_cols]
    df['image_name'] = df['image_name'].astype(str).str.strip()
    
    # Add prefix to columns
    df.columns = ['image_name'] + [f"{model_name}_{c}" for c in req_cols]
    
    return df

def load_and_standardize_pizza(filepath, model_name):
    """
    Loads the 'pizza' model CSV and calculates probabilities based on the 'final_prediction' column.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    if 'image_name' not in df.columns:
        print(f"WARNING: Model {model_name} missing 'image_name' column. Skipping.")
        return None
    
    # Required columns 
    req_cols = ['healthy', 'unhealthy', 'rubbish', 'final_prediction']
    if not all(c in df.columns for c in req_cols):
        print(f"WARNING: Model {model_name} missing one of {req_cols}. Skipping.")
        return None

    # Prepare new probability columns
    new_probs = {'Healthy': [], 'Unhealthy': [], 'Rubbish': []}
    
    for _, row in df.iterrows():
        label = row['final_prediction'].strip().lower()
        
        p_main = row[label]
        p_other = (1 - p_main) / 2
        
        if label == 'healthy':
            new_probs['Healthy'].append(p_main)
            new_probs['Unhealthy'].append(p_other)
            new_probs['Rubbish'].append(p_other)
        elif label == 'unhealthy':
            new_probs['Healthy'].append(p_other)
            new_probs['Unhealthy'].append(p_main)
            new_probs['Rubbish'].append(p_other)
        elif label == 'rubbish':
            new_probs['Healthy'].append(p_other)
            new_probs['Unhealthy'].append(p_other)
            new_probs['Rubbish'].append(p_main)
        else:
            # Handle cases with unexpected labels, distribute probability equally
            new_probs['Healthy'].append(1/3)
            new_probs['Unhealthy'].append(1/3)
            new_probs['Rubbish'].append(1/3)
            
    # Create the new dataframe
    out_df = pd.DataFrame({
        'image_name': df['image_name'].astype(str).str.strip(),
        f'{model_name}_Healthy': new_probs['Healthy'],
        f'{model_name}_Unhealthy': new_probs['Unhealthy'],
        f'{model_name}_Rubbish': new_probs['Rubbish']
    })
    
    return out_df

 #Load Data & Validate Overlaps
files = {
    "Model_predictions_isbi2025-ps3c-test-dataset": "predictions_isbi2025-ps3c-test-dataset.csv",
    "Model_probabilities_test_pizza": "final_prediction_test.csv",
    "Model_test_phase_prob": "test_phase_prob.csv", 
    "Model_Evaluation_Set": "Evaluation-set.csv",
    "Model_Tes_Set_ProbabilityScore": "Test_Set_ProbabilityScores.csv",
    "Model_isbi2025-ps3c-test-dataset pro Ens": "isbi2025-ps3c-test-dataset pro Ens.csv",
    #"Finetuned_model_Huina": "huina_new.csv",
    "Finetuned_model_Huina2":"validation_predictions2.csv",
    #"minine":"submission.csv",
}

ground_truth_file = "isbi2025-ps3c-test-dataset-annotated.csv"

print(f"Loading Ground Truth from {ground_truth_file}...")
gt_df = pd.read_csv(ground_truth_file)
gt_df = gt_df[['image_name', 'label']]
gt_df['label'] = gt_df['label'].str.lower().str.strip()
gt_df['image_name'] = gt_df['image_name'].str.strip()
gt_images = set(gt_df['image_name'].unique())

print(f"Ground Truth contains {len(gt_images)} images.")

dfs = []
valid_models = []

# Load models
for name, path in files.items():
    if name == "Model_probabilities_test_pizza":
        cleaned_df = load_and_standardize_pizza(path, name)
    else:
        is_4_class = (name == "Model_TestPhase")
        cleaned_df = load_and_standardize(path, name, is_four_class=is_4_class)
    
    if cleaned_df is not None:
        model_images = set(cleaned_df['image_name'].unique())
        overlap = gt_images.intersection(model_images)
        
        if len(overlap) == 0:
            print(f"warning: Model '{name}' has 0 overlapping images with Ground Truth. Excluding.")
        else:
            print(f" Model '{name}': {len(overlap)} overlapping images found. Included.")
            dfs.append(cleaned_df)
            valid_models.append(name)

if not dfs:
    print("Error: No models match the Ground Truth image names. Exiting.")
    sys.exit()

# 
# Merge Data
# 

merged_df = gt_df.copy()
for df in dfs:
    merged_df = pd.merge(merged_df, df, on='image_name', how='inner')

print(f"\nFinal Merged Dataset Shape: {merged_df.shape}")

if len(merged_df) == 0:
    print("Error: Merged dataset is empty. Check image names.")
    sys.exit()

#Prepare Ensembling

classes = ['Healthy', 'Unhealthy', 'Rubbish']
models = valid_models

def get_model_probs(df, model_name):
    cols = [f"{model_name}_{c}" for c in classes]
    return df[cols].values

label_map = {'healthy': 0, 'unhealthy': 1, 'rubbish': 2}
y_true = merged_df['label'].map(label_map).values

if np.isnan(y_true).any():
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    merged_df = merged_df[valid_mask]
    y_true = y_true.astype(int)
else:
    y_true = y_true.astype(int)

# 
# Ensemble Calculations
# 

# Storage for CV Statistics
cv_stats_list = []

def add_cv_stats(model_name, k, scores):
    cv_stats_list.append({
        "Model": model_name,
        "K-Folds": k,
        "Mean Accuracy": np.mean(scores),
        "Std Dev": np.std(scores)
    })

# A. Simple Average
print("\n--- Computing Simple Average ---")
avg_probs = np.zeros((len(merged_df), 3))
for m in models:
    avg_probs += get_model_probs(merged_df, m)
avg_probs /= len(models)
y_pred_avg = np.argmax(avg_probs, axis=1)

# B. Hard Voting
print("--- Computing Hard Voting ---")
votes = np.zeros((len(merged_df), 3))
for m in models:
    probs = get_model_probs(merged_df, m)
    preds = np.argmax(probs, axis=1)
    for i, p in enumerate(preds):
        votes[i, p] += 1
y_pred_voting = np.argmax(votes, axis=1)

# C. Weighted Average (CV Loop)
weighted_preds_variants = {}
weighted_probs_variants = {}

X_all = np.zeros((len(merged_df), len(models), 3))
for i, m in enumerate(models):
    X_all[:, i, :] = get_model_probs(merged_df, m)

for k in [5, 10]:
    print(f"--- Computing Weighted Average (Optimized) with {k}-Fold CV ---")
    weighted_probs_cv = np.zeros((len(merged_df), 3))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    wa_scores = []

    for train_idx, test_idx in skf.split(np.zeros(len(merged_df)), y_true):
        X_train, y_train = X_all[train_idx], y_true[train_idx]
        X_test, y_test = X_all[test_idx], y_true[test_idx]
        
        def loss_func_fold(weights):
            weights = np.array(weights)
            if np.sum(weights) == 0: return 100
            weights /= np.sum(weights)
            final_probs = np.sum(X_train * weights.reshape(1, -1, 1), axis=1)
            final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
            return log_loss(y_train, final_probs)

        init_weights = [1.0/len(models)] * len(models)
        bounds = [(0, 1)] * len(models)
        constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        
        try:
            res = minimize(loss_func_fold, init_weights, bounds=bounds, constraints=constraints)
            best_weights = res.x / np.sum(res.x)
        except:
            best_weights = init_weights

        fold_probs = np.sum(X_test * best_weights.reshape(1, -1, 1), axis=1)
        weighted_probs_cv[test_idx] = fold_probs
        wa_scores.append(accuracy_score(y_test, np.argmax(fold_probs, axis=1)))

    add_cv_stats("Weighted Average", k, wa_scores)
    weighted_probs_variants[k] = weighted_probs_cv
    weighted_preds_variants[k] = np.argmax(weighted_probs_cv, axis=1)

# D. Stacking (Linear)
stacking_lr_preds = {}
stacking_lr_probs = {}
X_stack = np.hstack([get_model_probs(merged_df, m) for m in models])
meta_model_lr = LogisticRegression(multi_class='multinomial', max_iter=1000)

for k in [5, 10]:
    print(f"--- Computing Stacking (Logistic) with {k}-Fold CV ---")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(meta_model_lr, X_stack, y_true, cv=skf, scoring='accuracy')
    add_cv_stats("Stacking (LogReg)", k, cv_scores)
    
    probs = cross_val_predict(meta_model_lr, X_stack, y_true, cv=skf, method='predict_proba')
    stacking_lr_probs[k] = probs
    stacking_lr_preds[k] = np.argmax(probs, axis=1)

# E. Rank Averaging
print("--- Computing Rank Averaging ---")
rank_accum = np.zeros((len(merged_df), 3))
for m in models:
    probs = get_model_probs(merged_df, m)
    for c in range(3):
        rank_accum[:, c] += rankdata(probs[:, c])
y_pred_rank = np.argmax(rank_accum, axis=1)

# F. Geometric Mean
print("--- Computing Geometric Mean ---")
stack_3d = np.array([get_model_probs(merged_df, m) for m in models])
stack_3d = np.moveaxis(stack_3d, 0, 2)
gmean_probs = gmean(stack_3d, axis=2)
y_pred_gmean = np.argmax(gmean_probs, axis=1)

# G. Stacking (Random Forest)
stacking_rf_preds = {}
stacking_rf_probs = {}
rf_meta = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

for k in [5, 10]:
    print(f"--- Computing Stacking (Random Forest) with {k}-Fold CV ---")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_meta, X_stack, y_true, cv=skf, scoring='accuracy')
    add_cv_stats("Stacking (RandomForest)", k, cv_scores)
    
    probs = cross_val_predict(rf_meta, X_stack, y_true, cv=skf, method='predict_proba')
    stacking_rf_probs[k] = probs
    stacking_rf_preds[k] = np.argmax(probs, axis=1)

# H. Stacking (Gradient Boosting)
stacking_gb_preds = {}
stacking_gb_probs = {}
gb_meta = GradientBoostingClassifier(n_estimators=116, max_depth=6, random_state=42)

for k in [5, 10]:
    print(f"--- Computing Stacking (Gradient Boosting) with {k}-Fold CV ---")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_meta, X_stack, y_true, cv=skf, scoring='accuracy')
    add_cv_stats("Stacking (GradBoost)", k, cv_scores)
    
    probs = cross_val_predict(gb_meta, X_stack, y_true, cv=skf, method='predict_proba')
    stacking_gb_probs[k] = probs
    stacking_gb_preds[k] = np.argmax(probs, axis=1)

# Evaluation & Metrics Collection

performance_data = []
all_preds = {}
def evaluate_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    all_preds[name] = y_pred
    print(f"{name: <35} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "F1-Score": f1}

print("\n=== Final Performance Metrics ===")

# Individual Models
for m in models:
    probs = get_model_probs(merged_df, m)
    preds = np.argmax(probs, axis=1)
    performance_data.append(evaluate_model(y_true, preds, m))

# Ensembles
performance_data.append(evaluate_model(y_true, y_pred_avg, "Simple Average"))
performance_data.append(evaluate_model(y_true, y_pred_voting, "Hard Voting"))
performance_data.append(evaluate_model(y_true, y_pred_rank, "Rank Averaging"))
performance_data.append(evaluate_model(y_true, y_pred_gmean, "Geometric Mean"))

# Add CV Variants
for k in [5, 10]:
    performance_data.append(evaluate_model(y_true, weighted_preds_variants[k], f"Weighted Average ({k}-Fold)"))
    performance_data.append(evaluate_model(y_true, stacking_lr_preds[k], f"Stacking LogReg ({k}-Fold)"))
    performance_data.append(evaluate_model(y_true, stacking_rf_preds[k], f"Stacking RF ({k}-Fold)"))
    performance_data.append(evaluate_model(y_true, stacking_gb_preds[k], f"Stacking GB ({k}-Fold)"))

# 
# Visualization
#

# 1. CV Statistics Table
cv_stats_df = pd.DataFrame(cv_stats_list)
print("\n=== Cross-Validation Statistics (Mean Accuracy +/- Std Dev) ===")
print(cv_stats_df)
cv_stats_df.to_csv("cv_statistics_summary.csv", index=False)
print("CV Statistics saved to cv_statistics_summary.csv")

# 2. Bar Chart
perf_df = pd.DataFrame(performance_data)
perf_melted = perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(18, 10), dpi =300)
sns.set_style("whitegrid")
chart = sns.barplot(data=perf_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Extended Ensemble Comparison (5-Fold and 10-Fold CV)", fontsize=16)
plt.ylim(0.5, 1.15)
plt.xticks(rotation=45, ha="right")
plt.legend(loc='lower right')
for container in chart.containers:
    chart.bar_label(container, fmt='%.4f', padding=3, fontsize=8, rotation=90)
plt.tight_layout()
plt.savefig("./Results/model_performance_chart_cv_comparison.png")
print(f"\nBar Chart saved to model_performance_chart_cv_comparison.png")

# Note: Detailed Confusion Matrix/ROC curves are skipped for brevity in this multi-fold comparison script,
# but you can use the 'all_preds' dictionary and 'stacking_lr_probs[10]' (for example) to generate them if needed.