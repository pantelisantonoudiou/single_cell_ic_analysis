# -*- coding: utf-8 -*-
# =============================================================================
#                                 Imports
# =============================================================================
import os
import numpy as np
import pandas as pd
from plots_with_stats import group_comparison_plot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
# =============================================================================

if __name__ == '__main__':

    # ========== USER CONFIGURATION ==========
    main_path  = r'R:\Pantelis\for analysis\patch_data_jamie\TRAP Ephys\full_dataset\analyzed'
    group_col = 'treatment'
    palette = ['#1f77b4', '#ff7f0e'] 
    n_cols    = 4
    # ========================================

    # ========== Load Data ==========
    path_basic_io  = os.path.join(main_path, 'io_basic', 'summary_io.csv')
    path_waveform  = os.path.join(main_path, 'io_wave',  'summary_waveform.csv')

    df_basic_io    = pd.read_csv(path_basic_io)
    df_waveform    = pd.read_csv(path_waveform)

    # ========== Define Columns ==========
    basic_io_cols = [
        'fr_at_20_percent_input', 'fr_at_40_percent_input', 'fr_at_60_percent_input',
        'fr_at_80_percent_input', 'fr_at_max_input', 'i_amp_at_half_max_fr',
        'input_resistance', 'resting_membrane_potential', 'max_firing_rate',
        'rheobase', 'io_slope'
    ]

    waveform_cols = [
        'ap_peak', 'threshold', 'ahp', 'peak_to_trough', 'rise_time', 'half_width'
    ]

    # ========== Plot and Stats ==========
    print("📊 Analyzing Basic/IO features...")
    res_basic_io = group_comparison_plot(
        df_basic_io,
        group_column=group_col,
        dependent_variables=basic_io_cols,
        palette=palette,
        n_cols=n_cols
    )

    print("📊 Analyzing Waveform features...")
    res_waveform = group_comparison_plot(
        df_waveform,
        group_column=group_col,
        dependent_variables=waveform_cols,
        palette=palette,
        n_cols=3
    )

    # # ========== Save Results ==========
    # res_basic_io.to_csv(os.path.join(main_path, 'stats_io_basic_io.csv'), index=False)
    # res_waveform.to_csv(os.path.join(main_path, 'stats_io_waveform.csv'), index=False)

    # print("✅ Comparison plots and stats saved.")
    
    # ==== COMBINE, PCA + GMM CLUSTERING & PLOT ====
    # 1) merge on cell_id & treatment
    df_merged = pd.merge(
        df_basic_io, df_waveform,
        on=['cell_id','treatment'], how='inner'
    )
    df_merged = df_merged.fillna(df_merged.median(numeric_only=True))

    # 2) pick features and standardize
    features = basic_io_cols + waveform_cols
    X = df_merged[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_merged[features].values)
    y = df_merged[group_col].values
    
    # 2. PCA to 2D for visualization & classification
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Stratified K-Fold Logistic Regression
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
    scores = []
    
    for train_idx, test_idx in skf.split(X_pca, y):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)
    
    print(f"✅ Mean CV Accuracy: {np.mean(scores):.3f}")
    print(f"📄 Fold Accuracies:  {np.round(scores, 3)}")
    
    # 4. Final model on full PCA data
    log_reg.fit(X_pca, y)
    
    # 5. Decision boundary plot
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 500),
        np.linspace(y_min, y_max, 500)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = log_reg.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    # 6. Scatter + decision boundary
    fig, ax = plt.subplots(figsize=(7, 7))
    for tr, col in zip(np.unique(y), palette):
        mask = y == tr
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=tr, c=col, edgecolor='k', s=50)
    
    ax.contour(xx, yy, Z, levels=[0.5], linestyles='--', colors='gray')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'PCA + Logistic Regression\n5-Fold CV Accuracy: {np.mean(scores):.2f}')
    ax.legend(title=group_col)
    plt.tight_layout()
    plt.show()
