# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import itertools
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
##### ------------------------------------------------------------------- #####

# Function to assign stars based on p-values
def stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'
    

def group_comparison_plot(data, group_column, dependent_variables, n_cols=3, palette='rocket',
                        p_adjustment_method='fdr_bh', stats_test='mannwhitneyu', show_plot=True):
    """
    This function takes a dataframe, group column, and a list of dependent variables,
    and returns a dataframe containing mean, standard deviation, SEM, Mann-Whitney U test statistic,
    p-values, and adjusted p-values for each dependent variable in both groups.
    It also generates bar plots for each variable.

    Parameters:
    data (pd.DataFrame): The input dataframe
    group_column (str): The name of the column containing group information
    dependent_variables (list of str): List of dependent variable column names
    n_cols (int): Number of columns in the plot grid
    palette (str): Seaborn color palette for the plots
    p_adjustment_method (str): Method for adjusting p-values for multiple comparisons
    stats_test (str): Statistical test function to use (default is Mann-Whitney U)
    show_plot (bool): Whether to display the plots

    Returns:
    results_df (pd.DataFrame): Dataframe containing mean, sd, SEM, Mann-Whitney U test statistic,
                                p-values, and adjusted p-values for each dependent variable in both groups
    """
    
    # Get stat test
    stat_func = getattr(sp_stats, stats_test)
    
    # Get groups
    group1, group2 = data[group_column].unique()
    
    # Initialize lists to store the results
    results = []
    
    # Run Wilcoxon rank-sum tests (Mann-Whitney U tests) and compute statistics for each dependent variable
    for var in dependent_variables:
        
        # get groups
        group1_data = data[data[group_column] == group1][var]
        group2_data = data[data[group_column] == group2][var]
        
        # drop nans
        group1_data = group1_data.dropna()
        group2_data = group2_data.dropna()
    
        # Calculate statistics
        mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
        sd1, sd2 = np.std(group1_data, ddof=1), np.std(group2_data, ddof=1)
        sem1, sem2 = sp_stats.sem(group1_data, nan_policy='omit'), sp_stats.sem(group2_data, nan_policy='omit')
        stat_test = stat_func(group1_data, group2_data, alternative='two-sided')
    
        results.append({
            'variable': var,
            'stats_table_report': '',  # Placeholder for the stats_report, will be updated later
            'stats_report': '',  # Placeholder for the stats_report, will be updated later
            f'{group1}_stats': f"{mean1:.2f} ± {sem1:.2f}",
            f'{group2}_stats': f"{mean2:.2f} ± {sem2:.2f}",
            f'{group1}_mean': mean1,
            f'{group1}_sd': sd1,
            f'{group1}_sem': sem1,
            f'{group2}_mean': mean2,
            f'{group2}_sd': sd2,
            f'{group2}_sem': sem2,
            f'{group1}_N': f"{group1_data.shape[0]}",
            f'{group2}_N': f"{group2_data.shape[0]}",
            'mann_whitney_u': stat_test.statistic,
            'p_value': stat_test.pvalue
        })
        
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        
        # Adjust p-values for multiple comparisons
        _, adjusted_p_values, _, _ = multipletests(results_df['p_value'], method=p_adjustment_method)
        results_df['adjusted_p_value'] = adjusted_p_values
        
        # Update the stats_table_report with adjusted p-value
        results_df['stats_table_report'] = results_df.apply(
            lambda row: f"U={row['mann_whitney_u']}, p={row['adjusted_p_value']:.3f}, N{group1}={row[f'{group1}_N']}, N{group2}={row[f'{group2}_N']}, ",
            axis=1
        )
        
        # Update the stats_report with adjusted p-value
        results_df['stats_report'] = results_df.apply(
            lambda row: f"{group1} = {row[f'{group1}_stats']}, {group2} = {row[f'{group2}_stats']}, N {group1} = {row[f'{group1}_N']}, N {group2} = {row[f'{group2}_N']}, Mann-Whitney, p = {row['adjusted_p_value']:.3f}",
            axis=1
        )

    if show_plot:
        
        # Create a long-format DataFrame for seaborn
        long_data = pd.melt(data, id_vars=group_column, value_vars=dependent_variables, var_name='variable', value_name='value')
        
        # Set up the subplots
        n_cols = n_cols
        n_rows = (len(dependent_variables) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), sharey=False)
        axs = axes.flatten()
        
        # Plot barplots for each variable and add stars
        for i, (var, adj_p_value) in enumerate(zip(dependent_variables, adjusted_p_values)):
            temp_data = long_data[long_data['variable'] == var]
            sns.stripplot(data=temp_data, x=group_column, y='value', ax=axs[i], hue=group_column, palette=palette, legend=False)
            sns.barplot(data=temp_data, x=group_column, y='value', ax=axs[i], errorbar='se', alpha=.5, palette=palette, hue=group_column, legend=False)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_title(f"{var} ({stars(adj_p_value)})")
        
        # Adjust layout and remove empty subplots if any
        fig.tight_layout()
        for idx in range(len(dependent_variables), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            fig.delaxes(axes[row, col])
        
    return results_df


def multi_group_comparison_plot(data, group_column, dependent_variables, n_cols=3, palette='rocket',
                        p_adjustment_method='fdr_bh', stats_test='mannwhitneyu', show_plot=True):
    
    # Get stat test
    stat_func = getattr(sp_stats, stats_test)
    
    # Get all unique groups
    groups = data[group_column].unique()
    if len(groups) < 2:
        print("Need at least two groups for comparison.")
        return None
    
    # Initialize lists to store the results
    results = []
    
    # Run statistical tests and compute statistics for each dependent variable
    for var in dependent_variables:
        
        # get all groups
        group_data = {group: data[data[group_column] == group][var].dropna() for group in groups}
        
        # Pairwise comparisons for each group with every other group
        for group1, group2 in itertools.combinations(groups, 2):
            
            mean1, mean2 = np.mean(group_data[group1]), np.mean(group_data[group2])
            sd1, sd2 = np.std(group_data[group1], ddof=1), np.std(group_data[group2], ddof=1)
            sem1, sem2 = sp_stats.sem(group_data[group1], nan_policy='omit'), sp_stats.sem(group_data[group2], nan_policy='omit')
            stat_test = stat_func(group_data[group1], group_data[group2], alternative='two-sided')
            
            results.append({
                'variable': var,
                'group1': group1,
                'group2': group2,
                f'{group1}_mean': mean1,
                f'{group1}_sd': sd1,
                f'{group1}_sem': sem1,
                f'{group2}_mean': mean2,
                f'{group2}_sd': sd2,
                f'{group2}_sem': sem2,
                f'{group1}_N': f"{group_data[group1].shape[0]}",
                f'{group2}_N': f"{group_data[group2].shape[0]}",
                'test_statistic': stat_test.statistic,
                'p_value': stat_test.pvalue
            })
                
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Adjust p-values for multiple comparisons
    _, adjusted_p_values, _, _ = multipletests(results_df['p_value'], method=p_adjustment_method)
    results_df['adjusted_p_value'] = adjusted_p_values
    
    # Update the stats_table_report with adjusted p-value
    results_df['stats_table_report'] = results_df.apply(
        lambda row: f"U={row['test_statistic']}, p={row['adjusted_p_value']:.3f}, N{row['group1']}={row[row['group1'] + '_N']}, N{row['group2']}={row[row['group2'] + '_N']} ",
        axis=1
    )
    if show_plot:
        # Create a long-format DataFrame for seaborn
        long_data = pd.melt(data, id_vars=group_column, value_vars=dependent_variables, var_name='variable', value_name='value')
        
        # Set up the subplots
        n_rows = (len(dependent_variables) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6), sharey=False)
        axs = axes.flatten()
        
        # Extract significant comparisons
        significant_comparisons = results_df[results_df['adjusted_p_value'] < 0.05]
        
        # Plot barplots for each variable
        for i, var in enumerate(dependent_variables):
            temp_data = long_data[long_data['variable'] == var]
            sns.stripplot(data=temp_data, x=group_column, y='value', ax=axs[i], hue=group_column, palette=palette, legend=False)
            sns.barplot(data=temp_data, x=group_column, y='value', ax=axs[i], errorbar='se', alpha=.5, palette=palette)
            
            # Annotate bars with significant p-values
            for index, row in significant_comparisons[significant_comparisons['variable'] == var].iterrows():
                group1, group2 = row['group1'], row['group2']
                # Get the x-tick positions based on the order of groups
                group1_index, group2_index = groups.tolist().index(group1), groups.tolist().index(group2)
                height = max(temp_data[temp_data[group_column].isin([group1, group2])]['value']) - 0.1  # some offset
                
                # Draw a horizontal line connecting the bars
                line_y = height + 0.02
                axs[i].plot([group1_index + 0.05, group2_index - 0.05], [line_y, line_y], color='black', lw=2)
                
                # Annotate with the p-value above the line
                center_x = (group1_index + group2_index) / 2
                axs[i].annotate(f"p={row['adjusted_p_value']:.3f}", (center_x, line_y + 0.01), ha='center', va='bottom', fontsize=14)
            
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_title(var)

        
        # Adjust layout and remove empty subplots if any
        fig.tight_layout()
        for idx in range(len(dependent_variables), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            fig.delaxes(axes[row, col])
        
    return results_df












