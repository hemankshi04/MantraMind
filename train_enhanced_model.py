"""
============================================================
  train_enhanced_model.py
  Mental Health Analysis — Enhanced Model Trainer
  GAN + Research-Based Dataset Integration

  HOW TO RUN:
    python train_enhanced_model.py

  WHAT IT DOES:
    Trains 6 ML models on 5 dataset combinations:
      1. Original research dataset        (200 rows)
      2. GAN-generated synthetic dataset  (200 rows)
      3. Original + GAN combined          (400 rows)
      4. Research paper extracted dataset (200 rows)
      5. All four datasets combined       (800 rows)

  OUTPUT:
    model/best_enhanced_model.pkl  ← ready for app.py
    outputs/fig1_accuracy_heatmap.png
    outputs/fig2_grouped_bar.png
    outputs/fig3_best_model_analysis.png
    outputs/fig4_gan_contribution.png
    outputs/fig5_feature_importance.png
    outputs/fig6_data_distributions.png
    outputs/fig7_dashboard.png
    outputs/results_summary.json
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, joblib, warnings, json
warnings.filterwarnings('ignore')

from sklearn.model_selection  import (train_test_split, cross_val_score,
                                       StratifiedKFold)
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       VotingClassifier)
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, f1_score,
                                       precision_score, recall_score)

print("=" * 70)
print("   MENTAL HEALTH ANALYSIS — ENHANCED MODEL WITH GAN + RESEARCH DATA")
print("=" * 70)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(BASE, 'dataset')
OUT    = os.path.join(BASE, 'outputs')
MDL    = os.path.join(BASE, 'model')
os.makedirs(OUT, exist_ok=True)
os.makedirs(MDL, exist_ok=True)

# ─── LOAD ALL DATASETS ────────────────────────────────────────────────────────
print("\n[1/8] Loading all datasets...")
df_orig   = pd.read_csv(os.path.join(DS_DIR, 'mantra_dataset.csv'))
df_gan    = pd.read_csv(os.path.join(DS_DIR, 'gan_mantra_dataset.csv'))
df_res_o  = pd.read_csv(os.path.join(DS_DIR, 'research_dataset_original.csv'))
df_res_g  = pd.read_csv(os.path.join(DS_DIR, 'research_dataset_generated.csv'))

df_combined = pd.concat([df_orig, df_gan], ignore_index=True)
df_all      = pd.concat([df_orig, df_gan, df_res_o, df_res_g], ignore_index=True)

datasets = {
    "Original (200)"       : df_orig,
    "GAN Synthetic (200)"  : df_gan,
    "Orig+GAN (400)"       : df_combined,
    "Research Paper (200)" : df_res_o,
    "All Combined (800)"   : df_all,
}

for name, df in datasets.items():
    dist = df['session_effectiveness'].value_counts().to_dict()
    print(f"  {name:<28} rows={len(df):<5} dist={dist}")

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
print("\n[2/8] Engineering features...")

def engineer_features(df):
    df = df.copy()
    df['stress_reduction']      = df['pre_stress']  - df['post_stress']
    df['anxiety_reduction']     = df['pre_anxiety'] - df['post_anxiety']
    df['focus_gain']            = df['post_focus']  - df['pre_focus']
    df['calm_gain']             = df['post_calm']   - df['pre_calm']
    df['total_improvement']     = (df['stress_reduction'] + df['anxiety_reduction']
                                   + df['focus_gain']     + df['calm_gain'])
    df['wellness_score']        = df['total_improvement'] + df['hrv_change']
    df['stress_anxiety_ratio']  = df['pre_stress'] / (df['pre_anxiety'] + 1)
    df['biofeedback_composite'] = (df['hrv_change'] + df['alpha_wave_increase']
                                   + df['theta_wave_increase']
                                   - df['cortisol_change_percent'])
    df['session_intensity']     = df['duration_minutes'] * df['repetitions_per_min']
    return df

for k in datasets:
    datasets[k] = engineer_features(datasets[k])
print("  Created 9 derived features ✓")

# ─── ENCODE ───────────────────────────────────────────────────────────────────
print("\n[3/8] Encoding categorical variables...")
le_mantra = LabelEncoder()
le_gender = LabelEncoder()
le_exp    = LabelEncoder()
le_target = LabelEncoder()

all_combined = pd.concat(list(datasets.values()))
le_mantra.fit(all_combined['mantra_type'])
le_gender.fit(all_combined['gender'])
le_exp.fit(all_combined['experience_level'])
le_target.fit(all_combined['session_effectiveness'])

def encode_df(df):
    df = df.copy()
    df['mantra_enc'] = le_mantra.transform(df['mantra_type'])
    df['gender_enc'] = le_gender.transform(df['gender'])
    df['exp_enc']    = le_exp.transform(df['experience_level'])
    df['target']     = le_target.transform(df['session_effectiveness'])
    return df

for k in datasets:
    datasets[k] = encode_df(datasets[k])

print(f"  Mantra classes : {list(le_mantra.classes_)}")
print(f"  Target classes : {list(le_target.classes_)}")

# ─── FEATURES LIST ────────────────────────────────────────────────────────────
FEATURES = [
    'mantra_enc','duration_minutes','repetitions_per_min','breath_sync_sec',
    'pre_stress','pre_anxiety','pre_focus','pre_calm',
    'hrv_change','cortisol_change_percent','alpha_wave_increase',
    'theta_wave_increase','age','gender_enc','exp_enc',
    'stress_reduction','anxiety_reduction','focus_gain','calm_gain',
    'total_improvement','wellness_score',
    'stress_anxiety_ratio','biofeedback_composite','session_intensity'
]

# ─── MODEL FACTORY ────────────────────────────────────────────────────────────
print("\n[4/8] Defining 6 models (RF, GBT, SVM, KNN, LR, Ensemble)...")

def make_models():
    rf  = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=42)
    svm = Pipeline([('sc', StandardScaler()),
                    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                                probability=True, random_state=42))])
    knn = Pipeline([('sc', StandardScaler()),
                    ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance'))])
    lr  = Pipeline([('sc', StandardScaler()),
                    ('lr',  LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
    ens = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft')
    return {
        'Random Forest'     : rf,
        'Gradient Boosting' : gb,
        'SVM'               : svm,
        'KNN'               : knn,
        'Logistic Reg'      : lr,
        'Voting Ensemble'   : ens,
    }

# ─── TRAIN & EVALUATE ─────────────────────────────────────────────────────────
print("\n[5/8] Training models across all 5 datasets...")
print("-" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

for ds_name, df in datasets.items():
    print(f"\n  Dataset: {ds_name} ({len(df)} rows)")
    X = df[FEATURES]
    y = df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                random_state=42, stratify=y)
    all_results[ds_name] = {}
    models = make_models()
    for mname, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        tr_acc = model.score(X_tr, y_tr)
        te_acc = accuracy_score(y_te, y_pred)
        cv_sc  = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        f1     = f1_score(y_te, y_pred, average='weighted')
        prec   = precision_score(y_te, y_pred, average='weighted')
        rec    = recall_score(y_te, y_pred, average='weighted')
        all_results[ds_name][mname] = {
            'model'    : model,
            'y_pred'   : y_pred,
            'y_test'   : y_te,
            'train_acc': tr_acc,
            'test_acc' : te_acc,
            'cv_mean'  : cv_sc.mean(),
            'cv_std'   : cv_sc.std(),
            'f1'       : f1,
            'precision': prec,
            'recall'   : rec,
        }
        tag = " ★" if te_acc >= 0.88 else (" ✓" if te_acc >= 0.80 else "")
        print(f"    {mname:<22} Train:{tr_acc*100:.1f}%  "
              f"Test:{te_acc*100:.1f}%  CV:{cv_sc.mean()*100:.1f}%"
              f"±{cv_sc.std()*100:.1f}%  F1:{f1:.3f}{tag}")

# ─── IDENTIFY BEST ────────────────────────────────────────────────────────────
print("\n[6/8] Identifying best model...")
best_acc = 0
bds, bmod = '', ''
for ds_name, mresults in all_results.items():
    for mname, m in mresults.items():
        if m['cv_mean'] > best_acc:
            best_acc = m['cv_mean']
            bds, bmod = ds_name, mname

best_res    = all_results[bds][bmod]
best_y_pred = best_res['y_pred']
best_y_test = best_res['y_test']
mod_labels  = list(list(all_results.values())[0].keys())
ds_labels   = list(all_results.keys())

print(f"  Best Dataset  : {bds}")
print(f"  Best Model    : {bmod}")
print(f"  CV Accuracy   : {best_res['cv_mean']*100:.2f}%")
print(f"  Test Accuracy : {best_res['test_acc']*100:.2f}%")
print(f"  F1 Score      : {best_res['f1']:.4f}")

# ─── SAVE MODEL & ENCODERS ────────────────────────────────────────────────────
print("\n[7/8] Saving best model artifacts...")
joblib.dump(best_res['model'],  os.path.join(MDL, 'best_model.pkl'))
joblib.dump(le_mantra,          os.path.join(MDL, 'le_mantra.pkl'))
joblib.dump(le_gender,          os.path.join(MDL, 'le_gender.pkl'))
joblib.dump(le_exp,             os.path.join(MDL, 'le_exp.pkl'))
joblib.dump(le_target,          os.path.join(MDL, 'le_target.pkl'))
joblib.dump(FEATURES,           os.path.join(MDL, 'feature_names.pkl'))
print("  ✓ model/best_model.pkl  (also compatible with app.py)")

# ─── CHARTS ──────────────────────────────────────────────────────────────────
print("\n[8/8] Generating 7 analysis charts...")

PALETTE = ['#2E4057','#048A81','#54C6EB','#EF946C','#C4A35A','#A44A3F']
BG      = '#F8F9FA'
ACCENT  = '#2E4057'
HIGH    = '#048A81'

# ── Fig 1: Accuracy Heatmap ──────────────────────────────────────────────────
matrix = np.zeros((len(mod_labels), len(ds_labels)))
for j, ds in enumerate(all_results):
    for i, mod in enumerate(all_results[ds]):
        matrix[i, j] = all_results[ds][mod]['cv_mean'] * 100

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
im = ax.imshow(matrix, cmap='YlGn', aspect='auto', vmin=60, vmax=100)
ax.set_xticks(range(len(ds_labels))); ax.set_yticks(range(len(mod_labels)))
ax.set_xticklabels(ds_labels, fontsize=10, fontweight='bold')
ax.set_yticklabels(mod_labels, fontsize=10)
for i in range(len(mod_labels)):
    for j in range(len(ds_labels)):
        v = matrix[i, j]
        ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=11,
                fontweight='bold', color='white' if v > 82 else '#1a1a2e')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label('CV Accuracy (%)', fontsize=10)
ax.set_title('Cross-Validation Accuracy (%) — All Models × All Datasets',
             fontsize=13, fontweight='bold', pad=14, color=ACCENT)
ax.set_xlabel('Dataset', fontsize=11); ax.set_ylabel('Model', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_accuracy_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig1_accuracy_heatmap.png")

# ── Fig 2: Grouped Bar ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
n_ds = len(ds_labels); width = 0.14; x = np.arange(len(mod_labels))
for j, (ds, col) in enumerate(zip(all_results, PALETTE)):
    vals   = [all_results[ds][m]['test_acc'] * 100 for m in mod_labels]
    offset = (j - n_ds/2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=ds, color=col, alpha=0.87,
           edgecolor='white', linewidth=0.5)
ax.axhline(80, color='red', ls='--', alpha=0.4, linewidth=1.2)
ax.axhline(90, color='green', ls='--', alpha=0.35, linewidth=1.2)
ax.text(len(mod_labels)-0.5, 80.5, '80% threshold', color='red', fontsize=8, alpha=0.7)
ax.text(len(mod_labels)-0.5, 90.5, '90% threshold', color='green', fontsize=8, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(mod_labels, fontsize=10)
ax.set_ylabel('Test Accuracy (%)'); ax.set_ylim(50, 108)
ax.set_title('Test Accuracy by Model and Dataset', fontsize=13, fontweight='bold', color=ACCENT)
ax.legend(loc='lower right', fontsize=8, framealpha=0.8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_grouped_bar.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig2_grouped_bar.png")

# ── Fig 3: Confusion Matrix + Class Report ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
for ax in axes: ax.set_facecolor(BG)
cm = confusion_matrix(best_y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_,
            linewidths=0.5, linecolor='white', annot_kws={'size':13, 'weight':'bold'})
axes[0].set_title(f'Confusion Matrix — {bmod}\n({bds})', fontsize=11, fontweight='bold', color=ACCENT)
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')
cr      = classification_report(best_y_test, best_y_pred,
                                 target_names=le_target.classes_, output_dict=True)
classes = le_target.classes_
x2      = np.arange(len(classes))
axes[1].bar(x2-0.25, [cr[c]['precision'] for c in classes], 0.25,
            label='Precision', color=PALETTE[0], alpha=0.85)
axes[1].bar(x2,      [cr[c]['recall']    for c in classes], 0.25,
            label='Recall',    color=PALETTE[1], alpha=0.85)
axes[1].bar(x2+0.25, [cr[c]['f1-score'] for c in classes], 0.25,
            label='F1-Score',  color=PALETTE[2], alpha=0.85)
axes[1].set_xticks(x2); axes[1].set_xticklabels(classes, fontsize=10)
axes[1].set_ylim(0, 1.15); axes[1].set_ylabel('Score')
axes[1].set_title('Precision / Recall / F1 by Class', fontsize=11, fontweight='bold', color=ACCENT)
axes[1].legend(fontsize=9)
for sp in ['top','right']: axes[1].spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_best_model_analysis.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig3_best_model_analysis.png")

# ── Fig 4: GAN Contribution Line Chart ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('GAN Contribution to Model Performance', fontsize=13, fontweight='bold', color=ACCENT, y=1.01)
for ax in axes: ax.set_facecolor(BG)
metrics_to_plot = [
    ('cv_mean',  'Cross-Val Accuracy (%)', True,  axes[0]),
    ('test_acc', 'Test Accuracy (%)',       True,  axes[1]),
    ('f1',       'Weighted F1-Score',       False, axes[2]),
]
for metric, title, pct, ax in metrics_to_plot:
    for i, mod in enumerate(mod_labels):
        vals = [all_results[ds][mod][metric] * (100 if pct else 1) for ds in ds_labels]
        ax.plot(range(len(ds_labels)), vals, marker='o', linewidth=2,
                alpha=0.8, label=mod, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(range(len(ds_labels)))
    ax.set_xticklabels(ds_labels, fontsize=8, rotation=15, ha='right')
    ax.set_title(title, fontsize=10, fontweight='bold', color=ACCENT)
    ax.set_ylabel(title, fontsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
axes[0].legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_gan_contribution.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig4_gan_contribution.png")

# ── Fig 5: Feature Importance ────────────────────────────────────────────────
best_rf_acc, best_rf_ds = 0, None
for ds in all_results:
    if all_results[ds]['Random Forest']['cv_mean'] > best_rf_acc:
        best_rf_acc = all_results[ds]['Random Forest']['cv_mean']
        best_rf_ds  = ds
rf_model    = all_results[best_rf_ds]['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
top15       = importances.tail(15)
fig, ax     = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
colors = [HIGH if v > importances.median() else PALETTE[4] for v in top15.values]
bars   = ax.barh(range(len(top15)), top15.values, color=colors, alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(top15))); ax.set_yticklabels(top15.index, fontsize=9)
ax.set_xlabel('Importance Score', fontsize=10)
ax.set_title(f'Top 15 Feature Importances — Random Forest\n(Dataset: {best_rf_ds})',
             fontsize=12, fontweight='bold', color=ACCENT)
for bar, val in zip(bars, top15.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig5_feature_importance.png")

# ── Fig 6: Data Distribution Comparison ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.patch.set_facecolor(BG)
fig.suptitle('Dataset Distribution Comparison — Key Features', fontsize=13,
             fontweight='bold', color=ACCENT)
compare_dfs = {
    'Original' : df_orig,
    'GAN'      : df_gan,
    'Research' : df_res_o,
}
cols_to_plot = ['pre_stress','pre_anxiety','alpha_wave_increase',
                'theta_wave_increase','hrv_change','cortisol_change_percent']
titles       = ['Pre-Stress Level','Pre-Anxiety Level','Alpha Wave Increase (%)',
                'Theta Wave Increase (%)','HRV Change','Cortisol Change (%)']
for ax, col, title in zip(axes.flat, cols_to_plot, titles):
    ax.set_facecolor(BG)
    for i, (lbl, dff) in enumerate(compare_dfs.items()):
        dff[col].hist(ax=ax, bins=25, alpha=0.55, color=PALETTE[i],
                      label=lbl, density=True)
    ax.set_title(title, fontsize=10, fontweight='bold', color=ACCENT)
    ax.set_xlabel(col, fontsize=8); ax.set_ylabel('Density', fontsize=8)
    ax.legend(fontsize=7)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_data_distributions.png'), dpi=150, bbox_inches='tight')
plt.close(); print("  ✓ fig6_data_distributions.png")

# ── Fig 7: Summary Dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

ax_t = fig.add_subplot(gs[0, :])
ax_t.axis('off')
ax_t.text(0.5, 0.70,
    'Mental Health Mantra Analysis — Model Performance Dashboard',
    ha='center', va='center', fontsize=16, fontweight='bold', color='white',
    transform=ax_t.transAxes)
ax_t.text(0.5, 0.22,
    f'Best: {bmod}  |  Dataset: {bds}  |  '
    f'CV: {best_res["cv_mean"]*100:.2f}%  |  '
    f'Test: {best_res["test_acc"]*100:.2f}%  |  '
    f'F1: {best_res["f1"]:.3f}',
    ha='center', va='center', fontsize=10, color='#54C6EB',
    transform=ax_t.transAxes)

card_data = [
    ('Original\nDataset', f"{all_results['Original (200)']['Voting Ensemble']['test_acc']*100:.1f}%", '#048A81'),
    ('+ GAN Data',        f"{all_results['Orig+GAN (400)']['Voting Ensemble']['test_acc']*100:.1f}%", '#EF946C'),
    ('All Data\nCombined',f"{all_results['All Combined (800)']['Voting Ensemble']['test_acc']*100:.1f}%",'#C4A35A'),
    ('Best F1\nScore',    f"{max(all_results[bds][m]['f1'] for m in mod_labels):.3f}", '#54C6EB'),
]
for idx, (label, val, col) in enumerate(card_data):
    axc = fig.add_subplot(gs[1, idx])
    axc.set_facecolor(col)
    axc.text(0.5, 0.62, val,   ha='center', va='center', fontsize=22,
             fontweight='bold', color='white', transform=axc.transAxes)
    axc.text(0.5, 0.22, label, ha='center', va='center', fontsize=9,
             color='white', transform=axc.transAxes)
    axc.set_xticks([]); axc.set_yticks([])

ax_tr = fig.add_subplot(gs[2, :2])
ax_tr.set_facecolor('#2a2a4a')
best_per_ds = [max(all_results[ds][m]['cv_mean'] for m in mod_labels)*100 for ds in ds_labels]
ax_tr.plot(range(len(ds_labels)), best_per_ds, 'o-', color='#54C6EB', linewidth=2.5, markersize=8)
for xi, yi in enumerate(best_per_ds):
    ax_tr.text(xi, yi+0.4, f'{yi:.1f}%', ha='center', fontsize=8, color='white', fontweight='bold')
ax_tr.set_xticks(range(len(ds_labels)))
ax_tr.set_xticklabels(ds_labels, fontsize=7, color='white', rotation=10, ha='right')
ax_tr.set_ylabel('Best CV Acc (%)', fontsize=9, color='white')
ax_tr.set_title('Best CV Accuracy Trend by Dataset', fontsize=10, fontweight='bold', color='white')
ax_tr.tick_params(colors='white')
for sp in ax_tr.spines.values(): sp.set_color('#444466')

ax_pi = fig.add_subplot(gs[2, 2:])
ax_pi.set_facecolor('#2a2a4a')
class_counts = df_all['session_effectiveness'].value_counts()
wedges, texts, autotexts = ax_pi.pie(
    class_counts.values, labels=class_counts.index,
    autopct='%1.1f%%', colors=PALETTE[:3], startangle=90,
    textprops={'fontsize':8, 'color':'white'},
    wedgeprops={'edgecolor':'#1a1a2e', 'linewidth':2})
for t in autotexts: t.set_color('white'); t.set_fontsize(9)
ax_pi.set_title('All-Data Class Distribution', fontsize=10, fontweight='bold', color='white')

plt.savefig(os.path.join(OUT, 'fig7_dashboard.png'), dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close(); print("  ✓ fig7_dashboard.png")

# ─── SAVE RESULTS JSON ────────────────────────────────────────────────────────
summary = {}
for ds in all_results:
    summary[ds] = {}
    for mod in all_results[ds]:
        r = all_results[ds][mod]
        summary[ds][mod] = {
            'train_acc': round(r['train_acc'], 4),
            'test_acc' : round(r['test_acc'],  4),
            'cv_mean'  : round(r['cv_mean'],   4),
            'cv_std'   : round(r['cv_std'],    4),
            'f1'       : round(r['f1'],        4),
            'precision': round(r['precision'], 4),
            'recall'   : round(r['recall'],    4),
        }
with open(os.path.join(OUT, 'results_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  ✓ results_summary.json")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ✅  TRAINING COMPLETE!")
print("=" * 70)
print(f"\n  🏆 BEST MODEL     : {bmod}")
print(f"  📊 BEST DATASET   : {bds}")
print(f"  🎯 CV ACCURACY    : {best_res['cv_mean']*100:.2f}% ± {best_res['cv_std']*100:.2f}%")
print(f"  🎯 TEST ACCURACY  : {best_res['test_acc']*100:.2f}%")
print(f"  📈 F1 SCORE       : {best_res['f1']:.4f}")
print(f"  📈 PRECISION      : {best_res['precision']:.4f}")
print(f"  📈 RECALL         : {best_res['recall']:.4f}")
print(f"\n  CLASSIFICATION REPORT:")
print(classification_report(best_y_test, best_y_pred, target_names=le_target.classes_))
print(f"  OUTPUT FILES → outputs/")
print("  ├─ fig1_accuracy_heatmap.png")
print("  ├─ fig2_grouped_bar.png")
print("  ├─ fig3_best_model_analysis.png")
print("  ├─ fig4_gan_contribution.png")
print("  ├─ fig5_feature_importance.png")
print("  ├─ fig6_data_distributions.png")
print("  ├─ fig7_dashboard.png")
print("  ├─ results_summary.json")
print("  └─ model/best_model.pkl  ← ready for app.py")
print("\n  NEXT STEP → Run:  python app.py")
print("=" * 70)
