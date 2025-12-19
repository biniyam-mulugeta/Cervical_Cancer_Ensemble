import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import label_binarize
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
    "Finetuned_model_Huina2":"validation_predictions9.csv",
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
            print(f"⚠️  WARNING: Model '{name}' has 0 overlapping images with Ground Truth. Excluding.")
        else:
            print(f"✅ Model '{name}': {len(overlap)} overlapping images found. Included.")
            dfs.append(cleaned_df)
            valid_models.append(name)

if not dfs:
    print("\n❌ ERROR: No models match the Ground Truth image names. Exiting.")
    sys.exit()

# ---------------------------------------------------------
# 3. Merge Data
# ---------------------------------------------------------

merged_df = gt_df.copy()
for df in dfs:
    merged_df = pd.merge(merged_df, df, on='image_name', how='inner')

print(f"\nFinal Merged Dataset Shape: {merged_df.shape}")

if len(merged_df) == 0:
    print("❌ ERROR: Merged dataset is empty. Check image names.")
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

# ---------------------------------------------------------
# 5. Ensemble Calculations
# ---------------------------------------------------------

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

# C. Weighted Average
print("--- Computing Weighted Average (Optimized) ---")
def loss_func(weights):
    weights = np.array(weights)
    if np.sum(weights) == 0: return 100
    weights /= np.sum(weights)
    final_probs = np.zeros((len(merged_df), 3))
    for i, m in enumerate(models):
        final_probs += weights[i] * get_model_probs(merged_df, m)
    final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
    return log_loss(y_true, final_probs)

init_weights = [1.0/len(models)] * len(models)
bounds = [(0, 1)] * len(models)
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
try:
    res = minimize(loss_func, init_weights, bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)
except:
    best_weights = init_weights # Fallback

weighted_probs = np.zeros((len(merged_df), 3))
for i, m in enumerate(models):
    weighted_probs += best_weights[i] * get_model_probs(merged_df, m)
y_pred_weighted = np.argmax(weighted_probs, axis=1)

# D. Stacking (Linear)
print("--- Computing Stacking (Logistic) ---")
X_stack = np.hstack([get_model_probs(merged_df, m) for m in models])
meta_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
meta_model.fit(X_stack, y_true)
stacking_probs = meta_model.predict_proba(X_stack)
y_pred_stacking = np.argmax(stacking_probs, axis=1)

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

# G. Stacking (Random Forest - Non-Linear)
print("--- Computing Stacking (Random Forest) ---")
rf_meta = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_meta.fit(X_stack, y_true)
rf_probs = rf_meta.predict_proba(X_stack)
y_pred_rf = np.argmax(rf_probs, axis=1)

# H. Stacking (Gradient Boosting - Non-Linear)
print("--- Computing Stacking (Gradient Boosting) ---")
gb_meta = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gb_meta.fit(X_stack, y_true)
gb_probs = gb_meta.predict_proba(X_stack)
y_pred_gb = np.argmax(gb_probs, axis=1)

# Evaluation & Metrics Collection

performance_data = []
all_preds = {}
def evaluate_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    all_preds[name] = y_pred
    print(f"{name: <25} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "F1-Score": f1}

print("\n=== Final Performance Metrics ===")
# Store scores for later selection
model_accs = {} 

# Individual Models
for m in models:
    probs = get_model_probs(merged_df, m)
    preds = np.argmax(probs, axis=1)
    res = evaluate_model(y_true, preds, m)
    model_accs[m] = res['Accuracy']
    performance_data.append(res)

performance_data.append(evaluate_model(y_true, y_pred_avg, "Simple Average"))
performance_data.append(evaluate_model(y_true, y_pred_voting, "Hard Voting"))
performance_data.append(evaluate_model(y_true, y_pred_weighted, "Weighted Average"))
performance_data.append(evaluate_model(y_true, y_pred_stacking, "Stacking (LogReg)"))
performance_data.append(evaluate_model(y_true, y_pred_rank, "Rank Averaging"))
performance_data.append(evaluate_model(y_true, y_pred_gmean, "Geometric Mean"))
performance_data.append(evaluate_model(y_true, y_pred_rf, "Stacking (RandomForest)"))
performance_data.append(evaluate_model(y_true, y_pred_gb, "Stacking (GradBoost)"))


# --- Store all probabilities for curve generation ---
all_probs = {
    "Simple Average": avg_probs,
    "Hard Voting": None, # Cannot generate curves from votes
    "Weighted Average": weighted_probs,
    "Stacking (LogReg)": stacking_probs,
    "Rank Averaging": None, # Cannot generate curves from ranks
    "Geometric Mean": gmean_probs,
    "Stacking (RandomForest)": rf_probs,
    "Stacking (GradBoost)": gb_probs,
}
for m in models:
    all_probs[m] = get_model_probs(merged_df, m)

# ---------------------------------------------------------
# 7. Visualization (Bar Chart, CM, Radar, ROC, PR)
# ---------------------------------------------------------

# --- 7.1 Bar Chart ---
perf_df = pd.DataFrame(performance_data)
# Find best model (use F1-score, handle potential ties by taking the first)
best_model_name = perf_df.loc[perf_df['F1-Score'].idxmax()]['Model']
print(f"\n🏆 Best performing model is '{best_model_name}' based on F1-Score.")

perf_melted = perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(16, 9))
sns.set_style("whitegrid")
chart = sns.barplot(data=perf_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Extended Ensemble Comparison", fontsize=16)
plt.ylim(0.5, 1.15)
plt.xticks(rotation=45, ha="right")
plt.legend(loc='lower right')
for container in chart.containers:
    chart.bar_label(container, fmt='%.4f', padding=3, fontsize=9, rotation=90)
plt.tight_layout()
plt.savefig("model_performance_chart.png")
print(f"\nBar Chart saved to model_performance_chart.png")

# --- 7.2 Confusion Matrix for Best Model ---
best_model_preds = all_preds.get(best_model_name)
if best_model_preds is not None:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, best_model_preds)
    labels = ['Healthy', 'Unhealthy', 'Rubbish']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_best_model.png")
    print(f"Confusion Matrix for best model saved to confusion_matrix_best_model.png")

# --- 7.3 Radar Chart (Robust Selection) ---
# Dynamically pick: The Best Individual Model + Simple Average + Best Overall Model
best_individual_model = max(model_accs, key=model_accs.get)
selected_models = list(set([best_individual_model, "Simple Average", best_model_name]))

print(f"Selected for Radar Chart: {selected_models}")

radar_data = []
for m_name in selected_models:
    preds = all_preds.get(m_name)
    if preds is None: continue
    
    radar_data.append({
        "Model": m_name,
        "Accuracy": accuracy_score(y_true, preds),
        "F1": f1_score(y_true, preds, average='weighted'),
        "Precision": precision_score(y_true, preds, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, preds, average='weighted', zero_division=0)
    })

if radar_data:
    radar_df = pd.DataFrame(radar_data)
    categories = list(radar_df.columns[1:])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9], ["0.6", "0.7", "0.8", "0.9"], color="grey", size=7)
    plt.ylim(0.5, 1.0)
    
    colors_list = ['b', 'r', 'g', 'c', 'm', 'y']
    for i, row in radar_df.iterrows():
        values = row[categories].values.flatten().tolist()
        values += [values[0]]
        color = colors_list[i % len(colors_list)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'], color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    plt.title("Radar Chart Comparison", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig("radar_chart_comparison.png")
    print("Radar Chart saved to radar_chart_comparison.png")


# --- 7.4 ROC and PR Curves for Best Model ---
best_model_probs = all_probs.get(best_model_name)

if best_model_probs is not None:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]
    class_names = ['Healthy', 'Unhealthy', 'Rubbish']

    # --- ROC Curve ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], best_model_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), best_model_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) for {best_model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_best_model.png")
    print(f"ROC Curve for best model saved to roc_curve_best_model.png")

    # --- PR Curve ---
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], best_model_probs[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], best_model_probs[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), best_model_probs.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, best_model_probs, average="micro")
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average PR (AP = {average_precision["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label=f'PR curve of class {class_names[i]} (AP = {average_precision[i]:0.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall (PR) Curve for {best_model_name}')
    plt.legend(loc="lower left")
    plt.savefig(f"pr_curve_best_model.png")
    print(f"PR Curve for best model saved to pr_curve_best_model.png")


# ---------------------------------------------------------
# 8. Save Detailed Predictions
# ---------------------------------------------------------

results_df = pd.DataFrame({
    'image_name': merged_df['image_name'],
    'True_Label': merged_df['label']
})

inv_map = {v: k for k, v in label_map.items()}

for m in models:
    probs = get_model_probs(merged_df, m)
    preds = np.argmax(probs, axis=1)
    results_df[f'Pred_{m}'] = [inv_map[p] for p in preds]

results_df['Pred_Simple_Average'] = [inv_map[p] for p in y_pred_avg]
results_df['Pred_Rank_Avg'] = [inv_map[p] for p in y_pred_rank]
results_df['Pred_Stacking_RF'] = [inv_map[p] for p in y_pred_rf]
results_df['Pred_Stacking_GB'] = [inv_map[p] for p in y_pred_gb]

output_filename = "ensemble_results_comparison.csv"
results_df.to_csv(output_filename, index=False)
print(f"Detailed CSV results saved to {output_filename}")