import numpy as np
import pickle
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def format_number(num):
    """
    Format numbers using scientific notation
    """
    if isinstance(num, (int, float)):
        if num == 0:
            return "0.000000e+00"
        elif isinstance(num, int) and abs(num) < 1000:  # Keep small integers as-is
            return str(num)
        else:
            return f"{float(num):.6e}"  # Use scientific notation for all floats
    return str(num)

def safe_pearsonr(x, y):
    """
    Safe version of Pearson correlation that handles errors
    """
    try:
        result = pearsonr(x, y)[0]
        return float(result)  # Ensure float type
    except ValueError:
        return np.nan

def safe_cosine_similarity(x, y):
    """
    Safe version of cosine similarity that handles errors
    """
    try:
        result = cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0]
        return float(result)  # Ensure float type
    except ValueError:
        return np.nan

def analyze_cross_modal_correlations(all_mice, all_mouse_dict):
    """
    Analyze cross-modal correlations 
    """
    all_paired_correlations = []
    all_unpaired_correlations = []
    per_mouse_stats = {}
    
    for mouse in all_mice:
        train_oracle_3val = all_mouse_dict[mouse]
        mouse_name = mouse.split("-")[0]
        key = 'train'
        
        mouse_paired_corrs = []
        mouse_unpaired_corrs = []
        
        files = list(train_oracle_3val[key].keys())
        
        for file in files:
            ca_data = train_oracle_3val[key][file]['ca_mean']
            video_data = train_oracle_3val[key][file]['video_mean']
            
            paired_corr = safe_pearsonr(ca_data.flatten(), video_data.flatten())
            if not np.isnan(paired_corr):
                mouse_paired_corrs.append(float(paired_corr))
                all_paired_correlations.append(float(paired_corr))
            
            for other_file in files:
                if other_file != file:
                    other_video_data = train_oracle_3val[key][other_file]['video_mean']
                    unpaired_corr = safe_pearsonr(ca_data.flatten(), other_video_data.flatten())
                    if not np.isnan(unpaired_corr):
                        mouse_unpaired_corrs.append(float(unpaired_corr))
                        all_unpaired_correlations.append(float(unpaired_corr))
        
        if mouse_paired_corrs and mouse_unpaired_corrs:
            t_stat, p_val = ttest_ind(mouse_paired_corrs, mouse_unpaired_corrs)
            per_mouse_stats[mouse_name] = {
                'paired_mean': float(np.mean(mouse_paired_corrs)),
                'paired_std': float(np.std(mouse_paired_corrs)),
                'paired_n': len(mouse_paired_corrs),
                'unpaired_mean': float(np.mean(mouse_unpaired_corrs)),
                'unpaired_std': float(np.std(mouse_unpaired_corrs)),
                'unpaired_n': len(mouse_unpaired_corrs),
                't_stat': float(t_stat),
                'p_value': float(p_val)
            }
    
    # Calculate global statistics
    t_stat, p_value = ttest_ind(all_paired_correlations, all_unpaired_correlations)
    mann_whitney_stat, mann_whitney_p = mannwhitneyu(all_paired_correlations, all_unpaired_correlations)
    cohens_d = float((np.mean(all_paired_correlations) - np.mean(all_unpaired_correlations)) / 
                     np.sqrt((np.std(all_paired_correlations) ** 2 + np.std(all_unpaired_correlations) ** 2) / 2))
    
    statistics = {
        'paired_mean': float(np.mean(all_paired_correlations)),
        'paired_std': float(np.std(all_paired_correlations)),
        'unpaired_mean': float(np.mean(all_unpaired_correlations)),
        'unpaired_std': float(np.std(all_unpaired_correlations)),
        't_statistic': float(t_stat),
        't_pvalue': float(p_value),
        'mann_whitney_statistic': float(mann_whitney_stat),
        'mann_whitney_pvalue': float(mann_whitney_p),
        'cohens_d': cohens_d,
        'per_mouse_stats': per_mouse_stats,
        'all_paired_correlations': [float(x) for x in all_paired_correlations],
        'all_unpaired_correlations': [float(x) for x in all_unpaired_correlations]
    }
    
    return statistics

if __name__ == '__main__':
    # Load data
    with open(f'./middle_state/all_rep.pkl', "rb") as tf:
        all_mouse_dict = pickle.load(tf)

    # Define train and test mice
    train_mice = ['dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
                'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',      
                'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',  
                'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20',
                'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20']

    test_mice = [
            'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce', 
            'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20'
    ]

    # Run analysis for both groups
    print("Analyzing training mice...")
    train_statistics = analyze_cross_modal_correlations(train_mice, all_mouse_dict)
    
    print("Analyzing testing mice...")
    test_statistics = analyze_cross_modal_correlations(test_mice, all_mouse_dict)
    
    # Combine results into one dictionary
    combined_results = {
        'train': train_statistics,
        'test': test_statistics
    }
    
    # Save all results to one pickle file
    with open('cross_modal_correlations.pkl', 'wb') as f:
        pickle.dump(combined_results, f)
    
    # Write combined analysis results to one text file
    with open('cross_modal_analysis_results.txt', 'w') as f:
        # Training mice section
        f.write("=" * 80 + "\n")
        f.write("TRAINING MICE CROSS-MODAL CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Write descriptive statistics
        f.write("Descriptive Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Paired Correlations:\n")
        f.write(f"  Mean: {format_number(train_statistics['paired_mean'])}\n")
        f.write(f"  Std:  {format_number(train_statistics['paired_std'])}\n")
        f.write(f"  N:    {len(train_statistics['all_paired_correlations'])}\n\n")
        
        f.write(f"Unpaired Correlations:\n")
        f.write(f"  Mean: {format_number(train_statistics['unpaired_mean'])}\n")
        f.write(f"  Std:  {format_number(train_statistics['unpaired_std'])}\n")
        f.write(f"  N:    {len(train_statistics['all_unpaired_correlations'])}\n\n")
        
        # Write statistical test results
        f.write("Statistical Tests:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Independent t-test:\n")
        f.write(f"  t-statistic: {format_number(train_statistics['t_statistic'])}\n")
        f.write(f"  p-value:     {format_number(train_statistics['t_pvalue'])}\n\n")
        
        f.write(f"Mann-Whitney U test:\n")
        f.write(f"  statistic:   {format_number(train_statistics['mann_whitney_statistic'])}\n")
        f.write(f"  p-value:     {format_number(train_statistics['mann_whitney_pvalue'])}\n\n")
        
        # Write effect size
        f.write("Effect Size:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Cohen's d: {format_number(train_statistics['cohens_d'])}\n\n")
        
        # Write per-mouse statistics
        f.write("Per-Mouse Statistics:\n")
        f.write("-" * 30 + "\n")
        for mouse, stats in train_statistics['per_mouse_stats'].items():
            f.write(f"{mouse}:\n")
            f.write(f"  Paired Correlations:\n")
            f.write(f"    Mean: {format_number(stats['paired_mean'])}\n")
            f.write(f"    Std:  {format_number(stats['paired_std'])}\n")
            f.write(f"    N:    {stats['paired_n']}\n")
            f.write(f"  Unpaired Correlations:\n")
            f.write(f"    Mean: {format_number(stats['unpaired_mean'])}\n")
            f.write(f"    Std:  {format_number(stats['unpaired_std'])}\n")
            f.write(f"    N:    {stats['unpaired_n']}\n")
            f.write(f"  T-test:\n")
            f.write(f"    t-stat: {format_number(stats['t_stat'])}\n")
            f.write(f"    p-value: {format_number(stats['p_value'])}\n\n")
        
        # Testing mice section
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("TESTING MICE CROSS-MODAL CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Write descriptive statistics
        f.write("Descriptive Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Paired Correlations:\n")
        f.write(f"  Mean: {format_number(test_statistics['paired_mean'])}\n")
        f.write(f"  Std:  {format_number(test_statistics['paired_std'])}\n")
        f.write(f"  N:    {len(test_statistics['all_paired_correlations'])}\n\n")
        
        f.write(f"Unpaired Correlations:\n")
        f.write(f"  Mean: {format_number(test_statistics['unpaired_mean'])}\n")
        f.write(f"  Std:  {format_number(test_statistics['unpaired_std'])}\n")
        f.write(f"  N:    {len(test_statistics['all_unpaired_correlations'])}\n\n")
        
        # Write statistical test results
        f.write("Statistical Tests:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Independent t-test:\n")
        f.write(f"  t-statistic: {format_number(test_statistics['t_statistic'])}\n")
        f.write(f"  p-value:     {format_number(test_statistics['t_pvalue'])}\n\n")
        
        f.write(f"Mann-Whitney U test:\n")
        f.write(f"  statistic:   {format_number(test_statistics['mann_whitney_statistic'])}\n")
        f.write(f"  p-value:     {format_number(test_statistics['mann_whitney_pvalue'])}\n\n")
        
        # Write effect size
        f.write("Effect Size:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Cohen's d: {format_number(test_statistics['cohens_d'])}\n\n")
        
        # Write per-mouse statistics
        f.write("Per-Mouse Statistics:\n")
        f.write("-" * 30 + "\n")
        for mouse, stats in test_statistics['per_mouse_stats'].items():
            f.write(f"{mouse}:\n")
            f.write(f"  Paired Correlations:\n")
            f.write(f"    Mean: {format_number(stats['paired_mean'])}\n")
            f.write(f"    Std:  {format_number(stats['paired_std'])}\n")
            f.write(f"    N:    {stats['paired_n']}\n")
            f.write(f"  Unpaired Correlations:\n")
            f.write(f"    Mean: {format_number(stats['unpaired_mean'])}\n")
            f.write(f"    Std:  {format_number(stats['unpaired_std'])}\n")
            f.write(f"    N:    {stats['unpaired_n']}\n")
            f.write(f"  T-test:\n")
            f.write(f"    t-stat: {format_number(stats['t_stat'])}\n")
            f.write(f"    p-value: {format_number(stats['p_value'])}\n\n")
    
    print("Analysis complete. All results written to cross_modal_analysis_results.txt")