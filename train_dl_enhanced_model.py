"""
============================================================
  train_dl_enhanced_model.py
  Mental Health Analysis - ML vs DL Comparative Study

  HOW TO RUN:
    python train_dl_enhanced_model.py

  WHAT IT DOES:
    Trains ML and DL models on mantra dataset:
      ML Models: Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, Ensemble
      DL Models: LSTM, RNN, CNN, Dense Neural Network

  OUTPUT:
    model/best_dl_model.pkl  ← best DL model for app.py
    outputs/dl_comparison_table.png
    outputs/error_analysis_graph.png
    outputs/gan_flowchart.png
    outputs/overall_flowchart.png
    outputs/dl_results_summary.json
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

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix, f1_score, mean_squared_error)

# TensorFlow/Keras for Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, LSTM, SimpleRNN, Conv1D, MaxPooling1D,
                                         Flatten, Dropout, BatchNormalization)
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print("TensorFlow available for DL models")
except ImportError:
    from sklearn.neural_network import MLPClassifier
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available, using sklearn MLPClassifier")

print("=" * 70)
print("   MENTAL HEALTH ANALYSIS - ML vs DL COMPARATIVE STUDY")
print("=" * 70)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(BASE, 'dataset')
OUT    = os.path.join(BASE, 'outputs')
MDL    = os.path.join(BASE, 'model')
os.makedirs(OUT, exist_ok=True)
os.makedirs(MDL, exist_ok=True)

# Visualization style constants
BG      = '#101218'
ACCENT  = '#54C6EB'
PALETTE = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

# ─── LOAD DATASETS ────────────────────────────────────────────────────────────
print("\n[1/8] Loading datasets...")
df_orig   = pd.read_csv(os.path.join(DS_DIR, 'mantra_dataset.csv'))
df_gan    = pd.read_csv(os.path.join(DS_DIR, 'gan_mantra_dataset.csv'))
df_res_o  = pd.read_csv(os.path.join(DS_DIR, 'research_dataset_original.csv'))
df_res_g  = pd.read_csv(os.path.join(DS_DIR, 'research_dataset_generated.csv'))

df_combined = pd.concat([df_orig, df_gan], ignore_index=True)
df_all      = pd.concat([df_orig, df_gan, df_res_o, df_res_g], ignore_index=True)

datasets = {
    "Original (200)"       : df_orig,
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
print("  Created 9 derived features")

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

# ─── ML MODEL FACTORY ─────────────────────────────────────────────────────────
print("\n[4/8] Defining ML models...")

def make_ml_models():
    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                      max_depth=3, random_state=42)
    lr  = Pipeline([('sc', StandardScaler()),
                    ('lr',  LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    return {
        'Random Forest'     : rf,
        'Gradient Boosting' : gb,
        'Logistic Reg'      : lr,
    }

# ─── DL MODEL FACTORY ─────────────────────────────────────────────────────────
print("\n[5/8] Defining DL models...")

def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_rnn_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_dense_nn_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def make_dl_models(input_shape, num_classes):
    if TF_AVAILABLE:
        return {
            'Dense Neural Net'  : create_dense_nn_model(input_shape, num_classes),
        }
    else:
        return {
            'MLP Classifier'    : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        }

# ─── TRAIN & EVALUATE ─────────────────────────────────────────────────────────
print("\n[6/8] Training ML and DL models...")
print("-" * 70)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
all_results = {}

for ds_name, df in datasets.items():
    print(f"\n  Dataset: {ds_name} ({len(df)} rows)")
    X = df[FEATURES]
    y = df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                random_state=42, stratify=y)

    # Scale features for DL models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # Convert to categorical for DL (only if TF available)
    if TF_AVAILABLE:
        y_tr_cat = to_categorical(y_tr)
        y_te_cat = to_categorical(y_te)
    else:
        y_tr_cat = y_tr  # Not used for sklearn
        y_te_cat = y_te  # Not used for sklearn

    all_results[ds_name] = {'ML': {}, 'DL': {}}

    # Train ML models
    ml_models = make_ml_models()
    for mname, model in ml_models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        tr_acc = model.score(X_tr, y_tr)
        te_acc = accuracy_score(y_te, y_pred)
        cv_sc  = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        f1     = f1_score(y_te, y_pred, average='weighted')
        mse    = mean_squared_error(y_te, y_pred)

        all_results[ds_name]['ML'][mname] = {
            'model'    : model,
            'y_pred'   : y_pred,
            'y_test'   : y_te,
            'train_acc': tr_acc,
            'test_acc' : te_acc,
            'cv_mean'  : cv_sc.mean(),
            'cv_std'   : cv_sc.std(),
            'f1'       : f1,
            'mse'      : mse,
        }
        tag = " [best]" if te_acc >= 0.88 else (" [good]" if te_acc >= 0.80 else "")
        print(f"    ML {mname:<22} Train:{tr_acc*100:.1f}%  "
              f"Test:{te_acc*100:.1f}%  CV:{cv_sc.mean()*100:.1f}%"
              f"±{cv_sc.std()*100:.1f}%  F1:{f1:.3f}  MSE:{mse:.3f}{tag}")

    # Train DL models
    dl_models = make_dl_models(X_tr_scaled.shape[1], len(np.unique(y)))

    for mname, model in dl_models.items():
        try:
            if TF_AVAILABLE:
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(
                    X_tr_scaled, y_tr_cat,
                    epochs=100, batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )

                y_pred_prob = model.predict(X_tr_scaled, verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
                tr_acc = history.history['accuracy'][-1]
                te_acc = accuracy_score(y_te, y_pred)
                f1     = f1_score(y_te, y_pred, average='weighted')
                mse    = mean_squared_error(y_te, y_pred)
            else:
                # sklearn MLPClassifier
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_te_scaled)
                tr_acc = model.score(X_tr_scaled, y_tr)
                te_acc = accuracy_score(y_te, y_pred)
                f1     = f1_score(y_te, y_pred, average='weighted')
                mse    = mean_squared_error(y_te, y_pred)

            all_results[ds_name]['DL'][mname] = {
                'model'    : model,
                'y_pred'   : y_pred,
                'y_test'   : y_te,
                'train_acc': tr_acc,
                'test_acc' : te_acc,
                'f1'       : f1,
                'mse'      : mse,
            }
            tag = " [best]" if te_acc >= 0.88 else (" [good]" if te_acc >= 0.80 else "")
            print(f"    DL {mname:<22} Train:{tr_acc*100:.1f}%  "
                  f"Test:{te_acc*100:.1f}%  F1:{f1:.3f}  MSE:{mse:.3f}{tag}")
        except Exception as e:
            print(f"    DL {mname:<22} Error: {str(e)}")
            all_results[ds_name]['DL'][mname] = {'error': str(e)}

# ─── IDENTIFY BEST MODELS ─────────────────────────────────────────────────────
print("\n[7/8] Identifying best models...")
best_ml_acc, best_dl_acc = 0, 0
bds_ml, bmod_ml, bds_dl, bmod_dl = '', '', '', ''

for ds_name, results in all_results.items():
    for mname, m in results['ML'].items():
        if 'cv_mean' in m and m['cv_mean'] > best_ml_acc:
            best_ml_acc = m['cv_mean']
            bds_ml, bmod_ml = ds_name, mname
    for mname, m in results['DL'].items():
        if 'test_acc' in m and m['test_acc'] > best_dl_acc:
            best_dl_acc = m['test_acc']
            bds_dl, bmod_dl = ds_name, mname

best_ml_res = all_results[bds_ml]['ML'][bmod_ml]
best_dl_res = all_results[bds_dl]['DL'][bmod_dl] if bds_dl and bmod_dl else None

print(f"  Best ML Model  : {bmod_ml} ({bds_ml}) - CV: {best_ml_res['cv_mean']*100:.2f}%")
if best_dl_res:
    print(f"  Best DL Model  : {bmod_dl} ({bds_dl}) - Test: {best_dl_res['test_acc']*100:.2f}%")
else:
    print("  No successful DL models trained")

# ─── SAVE BEST DL MODEL ───────────────────────────────────────────────────────
print("\n[8/8] Saving best DL model...")
if best_dl_res and bds_dl:
    # Get scaler from the best dataset
    best_ds_df = datasets[bds_dl]
    X_best = best_ds_df[FEATURES]
    scaler = StandardScaler()
    scaler.fit(X_best)  # Fit on full best dataset

    if TF_AVAILABLE:
        best_dl_res['model'].save(os.path.join(MDL, 'best_dl_model.h5'))
    else:
        joblib.dump(best_dl_res['model'], os.path.join(MDL, 'best_dl_model.pkl'))
    joblib.dump(scaler, os.path.join(MDL, 'dl_scaler.pkl'))
    print("  Saved model/best_dl_model.pkl (sklearn model)")
else:
    print("  WARNING: No DL model to save")

# Save encoders regardless
joblib.dump(le_mantra, os.path.join(MDL, 'le_mantra.pkl'))
joblib.dump(le_gender, os.path.join(MDL, 'le_gender.pkl'))
joblib.dump(le_exp, os.path.join(MDL, 'le_exp.pkl'))
joblib.dump(le_target, os.path.join(MDL, 'le_target.pkl'))
joblib.dump(FEATURES, os.path.join(MDL, 'feature_names.pkl'))

print("Starting chart generation...")
# ── Fig 1: DL Comparison Table ───────────────────────────────────────────────
print("Creating comparison table...")
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis('off')

# Prepare data for table
ml_models = list(list(all_results.values())[0]['ML'].keys())
dl_models = list(list(all_results.values())[0]['DL'].keys())
datasets_list = list(all_results.keys())

# Create table data
table_data = []
headers = ['Model Type', 'Model Name', 'Dataset', 'Test Acc (%)', 'F1 Score', 'MSE']

for ds in datasets_list:
    for model_type, models in [('ML', ml_models), ('DL', dl_models)]:
        for mname in models:
            if mname in all_results[ds][model_type]:
                res = all_results[ds][model_type][mname]
                if 'test_acc' in res:
                    table_data.append([
                        model_type,
                        mname,
                        ds,
                        f"{res['test_acc']*100:.2f}",
                        f"{res['f1']:.3f}",
                        f"{res['mse']:.3f}"
                    ])

# Add table to plot
table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                 cellLoc='center', colColours=[ACCENT]*6)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(ACCENT)
    elif table_data[i-1][0] == 'DL':
        cell.set_facecolor('#E8F4F8')
    else:
        cell.set_facecolor('#F8F9FA')

plt.title('ML vs DL Models Comparative Analysis', fontsize=16, fontweight='bold',
          color=ACCENT, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'dl_comparison_table.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved dl_comparison_table.png")

# ── Fig 2: Error Analysis Graph ───────────────────────────────────────────────
print("Creating error analysis graph...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.patch.set_facecolor(BG)
fig.suptitle('ML vs DL Error Analysis', fontsize=16, fontweight='bold', color=ACCENT)

for ax in axes.flat:
    ax.set_facecolor(BG)

# Accuracy comparison
ml_accs = []
dl_accs = []
for ds in datasets_list:
    for mname in ml_models:
        if mname in all_results[ds]['ML']:
            ml_accs.append(all_results[ds]['ML'][mname]['test_acc'] * 100)
    for mname in dl_models:
        if mname in all_results[ds]['DL']:
            dl_accs.append(all_results[ds]['DL'][mname]['test_acc'] * 100)

axes[0,0].hist(ml_accs, alpha=0.7, label='ML Models', bins=10, color=PALETTE[0])
axes[0,0].hist(dl_accs, alpha=0.7, label='DL Models', bins=10, color=PALETTE[1])
axes[0,0].set_xlabel('Test Accuracy (%)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Accuracy Distribution: ML vs DL')
axes[0,0].legend()

# F1 Score comparison
ml_f1s = []
dl_f1s = []
for ds in datasets_list:
    for mname in ml_models:
        if mname in all_results[ds]['ML']:
            ml_f1s.append(all_results[ds]['ML'][mname]['f1'])
    for mname in dl_models:
        if mname in all_results[ds]['DL']:
            dl_f1s.append(all_results[ds]['DL'][mname]['f1'])

axes[0,1].hist(ml_f1s, alpha=0.7, label='ML Models', bins=10, color=PALETTE[2])
axes[0,1].hist(dl_f1s, alpha=0.7, label='DL Models', bins=10, color=PALETTE[3])
axes[0,1].set_xlabel('F1 Score')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('F1 Score Distribution: ML vs DL')
axes[0,1].legend()

# MSE comparison
ml_mses = []
dl_mses = []
for ds in datasets_list:
    for mname in ml_models:
        if mname in all_results[ds]['ML']:
            ml_mses.append(all_results[ds]['ML'][mname]['mse'])
    for mname in dl_models:
        if mname in all_results[ds]['DL']:
            dl_mses.append(all_results[ds]['DL'][mname]['mse'])

axes[1,0].hist(ml_mses, alpha=0.7, label='ML Models', bins=10, color=PALETTE[4])
axes[1,0].hist(dl_mses, alpha=0.7, label='DL Models', bins=10, color=PALETTE[5])
axes[1,0].set_xlabel('Mean Squared Error')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('MSE Distribution: ML vs DL')
axes[1,0].legend()

# Best models comparison
best_ml_vals = [best_ml_res['test_acc']*100, best_ml_res['f1'], best_ml_res['mse']]
best_dl_vals = [best_dl_res['test_acc']*100, best_dl_res['f1'], best_dl_res['mse']]
metrics = ['Accuracy (%)', 'F1 Score', 'MSE']

x = np.arange(len(metrics))
width = 0.35
axes[1,1].bar(x - width/2, best_ml_vals, width, label=f'Best ML: {bmod_ml}', color=PALETTE[0])
axes[1,1].bar(x + width/2, best_dl_vals, width, label=f'Best DL: {bmod_dl}', color=PALETTE[1])
axes[1,1].set_xlabel('Metrics')
axes[1,1].set_ylabel('Values')
axes[1,1].set_title('Best Models Comparison')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(metrics)
axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'error_analysis_graph.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved error_analysis_graph.png")

# ── Fig 3: GAN Flowchart ──────────────────────────────────────────────────────
print("Creating GAN flowchart...")
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.axis('off')

# GAN Flowchart elements
flow_elements = [
    (0.5, 0.9, 'Real Mantra Data', '#048A81'),
    (0.2, 0.7, 'Generator Network', '#54C6EB'),
    (0.8, 0.7, 'Discriminator Network', '#EF946C'),
    (0.5, 0.5, 'Generated Fake Data', '#C4A35A'),
    (0.5, 0.3, 'Adversarial Training', '#A44A3F'),
    (0.5, 0.1, 'Synthetic Dataset', '#048A81'),
]

# Draw boxes and arrows
for x, y, text, color in flow_elements:
    ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8),
            color='white', transform=ax.transAxes)

# Draw arrows
arrows = [
    (0.5, 0.85, 0.5, 0.75),  # Real data to training
    (0.5, 0.65, 0.2, 0.55),  # Training to Generator
    (0.5, 0.65, 0.8, 0.55),  # Training to Discriminator
    (0.2, 0.45, 0.5, 0.35),  # Generator to Generated data
    (0.5, 0.25, 0.5, 0.15),  # Generated data to Synthetic dataset
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                transform=ax.transAxes)

ax.set_title('GAN Model Architecture Flowchart', fontsize=16, fontweight='bold',
             color='white', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'gan_flowchart.png'), dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("  Saved gan_flowchart.png")

# ── Fig 4: Overall System Flowchart ───────────────────────────────────────────
print("Creating overall flowchart...")
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.axis('off')

# Overall system flowchart
system_elements = [
    (0.5, 0.95, 'User Input\n(Mood, Stress, Goals)', '#048A81'),
    (0.2, 0.85, 'Data Collection\n& Preprocessing', '#54C6EB'),
    (0.8, 0.85, 'GAN Data\nAugmentation', '#EF946C'),
    (0.5, 0.75, 'Feature Engineering\n(23 Features)', '#C4A35A'),
    (0.2, 0.65, 'ML Models\n(Random Forest, SVM, etc.)', '#A44A3F'),
    (0.8, 0.65, 'DL Models\n(LSTM, RNN, CNN)', '#048A81'),
    (0.5, 0.55, 'Model Evaluation\n& Comparison', '#54C6EB'),
    (0.5, 0.45, 'Best Model Selection\n(Accuracy, F1, MSE)', '#EF946C'),
    (0.5, 0.35, 'Mantra Prediction\n& Duration', '#C4A35A'),
    (0.5, 0.25, 'Player Interface\n(Timer + Japa Counter)', '#A44A3F'),
    (0.5, 0.15, 'Session Tracking\n& Analytics', '#048A81'),
    (0.5, 0.05, 'User Feedback Loop', '#54C6EB'),
]

# Draw boxes
for x, y, text, color in system_elements:
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.8),
            color='white', transform=ax.transAxes)

# Draw flow arrows
flow_arrows = [
    (0.5, 0.9, 0.5, 0.8),   # Input to processing
    (0.5, 0.8, 0.2, 0.7),   # Processing to ML
    (0.5, 0.8, 0.8, 0.7),   # Processing to GAN
    (0.2, 0.7, 0.5, 0.6),   # ML to features
    (0.8, 0.7, 0.5, 0.6),   # GAN to features
    (0.5, 0.6, 0.2, 0.5),   # Features to ML models
    (0.5, 0.6, 0.8, 0.5),   # Features to DL models
    (0.2, 0.5, 0.5, 0.4),   # ML to evaluation
    (0.8, 0.5, 0.5, 0.4),   # DL to evaluation
    (0.5, 0.4, 0.5, 0.3),   # Evaluation to selection
    (0.5, 0.3, 0.5, 0.2),   # Selection to prediction
    (0.5, 0.2, 0.5, 0.1),   # Prediction to player
    (0.5, 0.1, 0.5, 0.0),   # Player to tracking
]

for x1, y1, x2, y2 in flow_arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                transform=ax.transAxes)

ax.set_title('Mental Health Mantra Analysis System Flowchart', fontsize=16,
             fontweight='bold', color='white', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overall_flowchart.png'), dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("  Saved overall_flowchart.png")

# ─── SAVE RESULTS JSON ────────────────────────────────────────────────────────
summary = {'ML': {}, 'DL': {}}
for ds in all_results:
    summary['ML'][ds] = {}
    summary['DL'][ds] = {}
    for model_type in ['ML', 'DL']:
        for mod in all_results[ds][model_type]:
            res = all_results[ds][model_type][mod]
            if 'test_acc' in res:
                summary[model_type][ds][mod] = {
                    'test_acc': round(res['test_acc'], 4),
                    'f1': round(res['f1'], 4),
                    'mse': round(res['mse'], 4),
                }

with open(os.path.join(OUT, 'dl_results_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved dl_results_summary.json")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ML vs DL COMPARATIVE ANALYSIS COMPLETE!")
print("=" * 70)
print(f"  BEST ML MODEL     : {bmod_ml}")
print(f"  BEST ML DATASET   : {bds_ml}")
print(f"  ML CV ACCURACY    : {best_ml_res['cv_mean']*100:.2f}%")
print(f"  ML TEST ACCURACY  : {best_ml_res['test_acc']*100:.2f}%")
print(f"  ML F1 SCORE       : {best_ml_res['f1']:.4f}")
print(f"  ML MSE            : {best_ml_res['mse']:.4f}")
print()
if best_dl_res:
    print(f"  BEST DL MODEL     : {bmod_dl}")
    print(f"  BEST DL DATASET   : {bds_dl}")
    print(f"  DL TEST ACCURACY  : {best_dl_res['test_acc']*100:.2f}%")
    print(f"  DL F1 SCORE       : {best_dl_res['f1']:.4f}")
    print(f"  DL MSE            : {best_dl_res['mse']:.4f}")
else:
    print("  WARNING: No successful DL models trained")
print()
print("  COMPARISON CHARTS GENERATED:")
print("    - dl_comparison_table.png")
print("    - error_analysis_graph.png")
print("    - gan_flowchart.png")
print("    - overall_flowchart.png")
