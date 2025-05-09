import numpy as np
import pickle
from scipy.stats import pearsonr, ttest_ind, f_oneway, mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import multiprocessing as mp
import statsmodels.api as sm
from statsmodels.formula.api import ols

def safe_pearsonr(x, y):
    try:
        return pearsonr(x, y)[0]
    except ValueError:
        return np.nan

def safe_cosine_similarity(x, y):
    try:
        return cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))[0][0]
    except ValueError:
        return np.nan

def cohens_d(x, y):
    """
    Calculate Cohen's d effect size between two samples
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def process_mouse_pair(mouse, cup_mouse, all_mouse_dict):
    """
    Process a specific pair of mice for level 1 analysis.
    Compares paired and unpaired trials between the two mice.
    """
    train_oracle_3val = all_mouse_dict[mouse]
    train_oracle_3val_cup = all_mouse_dict[cup_mouse]
    
    mouse_data = []
    
    for lv_embed in ['ca_mean']:
        key = 'train'
        mouse_name = mouse.split("-")[0]
        cup_mouse_name = cup_mouse.split("-")[0]
        
        with open(f'./mapping_files/{mouse_name}_{cup_mouse_name}_CrossIndividuals.txt','r') as f:
            coupled_files = f.readlines()
        
        all_cos_paired = []
        all_corr_paired = []
        all_cos_unpaired = []
        all_corr_unpaired = []
        
        # paired
        for repeated_file in coupled_files:
            repeated_file = [file.split("_")[-1].split("\n")[0] for file in repeated_file.split(",")]
            data1 = train_oracle_3val[key][repeated_file[0]+'.pt'][lv_embed]
            data2 = train_oracle_3val_cup[key][repeated_file[1]+'.pt'][lv_embed]
            all_corr_paired.append(pearsonr(data1.flatten(), data2.flatten()).statistic)
            all_cos_paired.append(cosine_similarity(data1.reshape(1,-1), data2.reshape(1,-1))[0][0])
        
        # unpaired
        files1 = list(train_oracle_3val[key].keys())
        files2 = list(train_oracle_3val_cup[key].keys())
        coupled_pairs = set((f.split(",")[0].split("_")[-1], f.split(",")[1].split("_")[-1].strip()) for f in coupled_files)
        
        for file1 in files1:
            for file2 in files2:
                if (file1.split('.')[0], file2.split('.')[0]) not in coupled_pairs:
                    data1 = train_oracle_3val[key][file1][lv_embed]
                    data2 = train_oracle_3val_cup[key][file2][lv_embed]
                    all_corr_unpaired.append(pearsonr(data1.flatten(), data2.flatten()).statistic)
                    all_cos_unpaired.append(cosine_similarity(data1.reshape(1,-1), data2.reshape(1,-1))[0][0])
        
        mouse_data.extend([
            {'mouse_pair': f'{mouse_name}_{cup_mouse_name}', 'embedding': lv_embed, 'paired': 'paired', 'R': r, 'cosine': c}
            for r, c in zip(all_corr_paired, all_cos_paired)
        ])
        mouse_data.extend([
            {'mouse_pair': f'{mouse_name}_{cup_mouse_name}', 'embedding': lv_embed, 'paired': 'unpaired', 'R': r, 'cosine': c}
            for r, c in zip(all_corr_unpaired, all_cos_unpaired)
        ])
    
    return mouse_data

def process_analysis(sub_mice, cup_mice, all_mouse_dict, cpu_limit='auto'):
    """
    Process level 1 analysis for specific pairs of mice.
    """
    if cpu_limit == 'auto':
        num_cpus = max(1, round(mp.cpu_count() * 0.75))
    elif isinstance(cpu_limit, float) and 0 < cpu_limit < 1:
        num_cpus = max(1, round(mp.cpu_count() * cpu_limit))
    elif isinstance(cpu_limit, int) and cpu_limit > 0:
        num_cpus = min(cpu_limit, mp.cpu_count())
    else:
        num_cpus = max(1, mp.cpu_count() - 2)

    with mp.Pool(num_cpus) as pool:
        results = pool.starmap(process_mouse_pair, [(mouse, cup_mouse, all_mouse_dict) for mouse, cup_mouse in zip(sub_mice, cup_mice)])
    stats_results = [item for sublist in results for item in sublist]
    return stats_results, pd.DataFrame(stats_results)

def format_value(value, precision=6):
    """
    Format values, using scientific notation for all numbers less than 1
    """
    if isinstance(value, (int, float)):
        if value == 0:
            return "0.000000e+00"
        elif abs(value) < 1 or abs(value) > 1e4:
            return f"{value:.{precision}e}"  # Use scientific notation
        else:
            return f"{value:.{precision}f}"  # Keep specified decimal places
    return str(value)

def format_number(num):
    """
    Format numbers with maximum precision using scientific notation
    """
    if isinstance(num, (int, float)):
        if num == 0:
            return "0.000000e+00"
        elif isinstance(num, int) and abs(num) < 1000:
            return str(num)
        else:
            # For very small p-values, use higher precision scientific notation
            return f"{num:.16e}"  # Use 16 digits precision
    return str(num)

def run_statistical_analysis_analysis(df):
    """
    Run statistical analysis for level 1 data
    """
    results = {}
    
    for embed_type in ['ca_mean']:
        embed_df = df[df['embedding'] == embed_type]
        
        results[f"{embed_type} Analysis"] = {}
        
        # 1. Independent t-test with full precision
        paired_r = embed_df[embed_df['paired'] == 'paired']['R']
        unpaired_r = embed_df[embed_df['paired'] == 'unpaired']['R']
        t_stat, p_value = ttest_ind(paired_r, unpaired_r)
        results[f"{embed_type} Analysis"]['Independent t-test'] = {
            't-statistic': format_number(float(t_stat)), 
            'p-value': format_number(float(p_value))
        }

        # 2. Mann-Whitney U test with full precision
        statistic, p_value = mannwhitneyu(paired_r, unpaired_r)
        results[f"{embed_type} Analysis"]['Mann-Whitney U test'] = {
            'statistic': format_number(float(statistic)), 
            'p-value': format_number(float(p_value))
        }

        # 3. One-way ANOVA with full precision
        groups = [group['R'].values for name, group in embed_df.groupby('mouse_pair')]
        f_stat, p_value = f_oneway(*groups)
        results[f"{embed_type} Analysis"]['One-way ANOVA'] = {
            'F-statistic': format_number(float(f_stat)), 
            'p-value': format_number(float(p_value))
        }

        # 4. Two-way ANOVA
        model = ols('R ~ C(mouse_pair) + C(paired) + C(mouse_pair):C(paired)', data=embed_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        # Convert all values in anova_table to use high precision format
        anova_table = anova_table.applymap(lambda x: format_number(float(x)) if isinstance(x, (int, float)) else x)
        results[f"{embed_type} Analysis"]['Two-way ANOVA'] = anova_table

        # 5. Effect size (Cohen's d)
        cohens_d = (np.mean(paired_r) - np.mean(unpaired_r)) / np.sqrt((np.std(paired_r, ddof=1) ** 2 + np.std(unpaired_r, ddof=1) ** 2) / 2)
        results[f"{embed_type} Analysis"]['Effect size (Cohen\'s d)'] = {
            'Cohen\'s d': format_number(float(cohens_d))
        }

        # 6. Descriptive statistics with full precision
        desc_stats = embed_df.groupby('paired')['R'].describe()
        desc_stats = desc_stats.applymap(lambda x: format_number(float(x)) if isinstance(x, (int, float)) else x)
        results[f"{embed_type} Analysis"]['Descriptive statistics'] = desc_stats

    return results

def write_full_dataframe(file, df):
    """
    Write a dataframe to file with formatted values and proper alignment
    """
    str_df = df.applymap(lambda x: format_value(x))
    
    max_widths = [max(str_df[col].map(len).max(), len(col)) for col in df.columns]
    header = " | ".join(col.ljust(width) for col, width in zip(df.columns, max_widths))
    separator = "-+-".join("-" * width for width in max_widths)
    file.write(f"{header}\n{separator}\n")
    for _, row in str_df.iterrows():
        file.write(" | ".join(val.ljust(width) for val, width in zip(row, max_widths)))
        file.write("\n")

def write_dict(file, d, indent=""):
    """
    Recursively write a dictionary to file with proper indentation
    """
    for key, value in d.items():
        file.write(f"{indent}{key}:\n")
        if isinstance(value, dict):
            write_dict(file, value, indent + "  ")
        elif isinstance(value, pd.DataFrame):
            file.write(indent + "  ")
            write_full_dataframe(file, value)
        else:
            file.write(f"{indent}  {format_value(value)}\n")

if __name__ == '__main__':
    with open(f'./middle_state/all_rep.pkl', "rb") as tf:
        all_mouse_dict = pickle.load(tf)

    sub_mice = [#'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
            'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce', # test mouse for verifying zero-shot cross-subject preserved neural representation
            #'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
            #'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
            #'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    ]
    cup_mice = [#'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20', 
            'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20', # test mouse
            #'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',  
            #'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20', 
            #'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20'
    ]  
    
    stats_results, stats_df = process_analysis(sub_mice, cup_mice, all_mouse_dict)
    
    # Save results
    with open('neural_activtiy_analysis.pkl', 'wb') as f:
        pickle.dump(stats_results, f)

    print("DataFrame Info:")
    print(stats_df.info())
    print("\nValue counts for 'embedding' column:")
    print(stats_df['embedding'].value_counts())
    print("\nValue counts for 'paired' column:")
    print(stats_df['paired'].value_counts())
    print("\nSummary statistics:")
    print(stats_df.groupby(['embedding', 'paired'])[['R', 'cosine']].describe())
    
    # Run statistical analysis 
    analysis_results = run_statistical_analysis_analysis(stats_df)
    with open('./similarity_comparison.txt', 'w') as write_f:
        for test_name, result in analysis_results.items():
            write_f.write(f"{'=' * 80}\n")
            write_f.write(f"{test_name}\n")
            write_f.write(f"{'=' * 80}\n")
            if isinstance(result, pd.DataFrame):
                write_full_dataframe(write_f, result)
            elif isinstance(result, dict):
                write_dict(write_f, result)
            else:
                write_f.write(format_value(result))
            write_f.write("\n\n")
    
    print("Analysis complete. Results written to similarity_comparison.txt")

