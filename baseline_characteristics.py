import pickle
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from datetime import date
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu

with open('datasets/pat_data_rev.pkl', 'rb') as handle:
    pat_baseline = pickle.load(handle)

concept = pd.read_csv('datasets/concept.csv')
concept_dict = dict(zip(concept['concept_id'], concept['concept_name']))

test_ids = {}
for months_prior in tqdm(['3-6m', '6-12m', '12-36m']):
    pat_data_path = 'output/baseline_binary1536_'+str(months_prior)+'.pickle'
    
    with open(pat_data_path, 'rb') as handle:
        test_ids[months_prior] = pickle.load(handle)

with open('datasets/pc_diag.pickle', 'rb') as h:
    pc_diag = pickle.load(h)

labels = ['3-6m', '6-12m', '12-36m']
with_and_without_rf = {}
print('num pc_with_rf vs pc_without_rf')
for months_prior in labels:
    s = months_prior.split('-')[0]
    e = months_prior.split('-')[1].split('m')[0]
    gaps = [(int(s), int(e))]

    before = defaultdict(list)
    after = []
    indices = [i for i, x in enumerate(test_ids[months_prior]['labels']) if x == 1]
    case = [test_ids[months_prior]['ids'][i] for i in indices]
    pc_with_no_rf = []
    for i in ['pancreatitis', 'type2diabetes', 'CA19']:
    # i = 'pancreatitis'
        rf= pd.read_csv('datasets/'+i+'.csv')
        rf.columns = ['person_id', 'date', 'concept_id']
        rf['date'] = pd.to_datetime(rf['date'], errors = 'coerce')

        rf['sorted_date'] = rf.sort_values(by='date').groupby('person_id')['date'].transform('first')
        rf_dict = rf.drop_duplicates(subset='person_id').set_index('person_id')['sorted_date'].to_dict()

        for k in case:  # No need to convert to a list
            if k in list(rf_dict.keys()):
                diff = pc_diag[k] - rf_dict[k]
                for gap in gaps:
                    # Check if the difference in days falls within the current gap
                    # if gap[0] * 30 <= diff.days <= gap[1] * 30:
                    if gap[0] * 30 <= diff.days:
                        before[gap].append(k)
                        break
                else:  # Only execute if the loop wasn't broken (i.e., diff.days < 0)
                    if diff.days < 0:
                        after.append(k)
            else:
                pc_with_no_rf.append(k)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(months_prior, file=f)
        print('case number:', len(case), file=f)
        print('pc_with_no_rf:',len(set(pc_with_no_rf)), file=f)
        print('with RF:', len(set(before[gap])), file=f)
        print('RF later:', len(set(after)), file=f)

        pc_with_rf = list(set(before[gap]))
        pc_with_no_rf = list(set(case)-set(before[gap]))
        print('without RF:', len(pc_with_no_rf), file=f)
        print('----------------------------', file=f)

    comorb = {}
    ages = {}
    gender_lst = {}
    eth_lst = {}
    race_lst = {}
    age_lst = {}

    for i in ['with_rf','without_rf']:
        comorb[i] = []
        ages[i] = []
        gender_lst[i] = []
        eth_lst[i] = []
        race_lst[i] = []
        age_lst[i] = []

    today = pd.to_datetime(date.today())
    for i in pc_with_rf:
        birth_date = pat_baseline[i]['birth_date']
        comorb['with_rf'].extend(list(set(pat_baseline[i]['concept_dx'])))
        diag_date = pat_baseline[i]['age_at_diagnosis']
        age_at_diag = (diag_date - birth_date).days / 365.25
        ages['with_rf'].append(age_at_diag)
        gender_lst['with_rf'].append(pat_baseline[i]['gender'])
        eth_lst['with_rf'].append(pat_baseline[i]['ethnicity'])
        race_lst['with_rf'].append(pat_baseline[i]['race'])
        age_lst['with_rf'].append(pat_baseline[i]['age'])
        # age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))

    for i in pc_with_no_rf:
        birth_date = pat_baseline[i]['birth_date']
        comorb['without_rf'].extend(list(set(pat_baseline[i]['concept_dx'])))
        diag_date = pat_baseline[i]['age_at_diagnosis']
        age_at_diag = (diag_date - birth_date).days / 365.25
        ages['without_rf'].append(age_at_diag)
        gender_lst['without_rf'].append(pat_baseline[i]['gender'])
        eth_lst['without_rf'].append(pat_baseline[i]['ethnicity'])
        race_lst['without_rf'].append(pat_baseline[i]['race'])
        age_lst['without_rf'].append(pat_baseline[i]['age'])

    with_and_without_rf[months_prior] = [comorb, ages, gender_lst, eth_lst, race_lst, age_lst]

def ci95(data):
    return np.percentile(data, [2.5, 97.5])
    
    # Example: two lists of age distributions for 3 time windows
group1, group2 = [], []
for months_prior in labels:
    group1.append(with_and_without_rf[months_prior][1]['with_rf'])
    group2.append(with_and_without_rf[months_prior][1]['without_rf'])

# Combine for grouped boxplot
data = []
positions = []
print('compare age at diagnosis')
for i in range(3):
    data += [group1[i], group2[i]]
    positions += [i*3 + 1, i*3 + 2]  # space between groups

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(data, positions=positions, widths=0.6)

# Set x-axis
xtick_positions = [i*3 + 1.5 for i in range(3)]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(labels, fontsize = 12)
ax.set_ylabel('Age', fontsize = 12)
ax.legend([plt.Line2D([0], [0], color='black')], ['with RF vs without RF'], fontsize = 12)

for i in range(3):
    g1 = group1[i]
    g2 = group2[i]
    
    # T-test
    # t_stat, p_val = ttest_ind(g1, g2, equal_var=False)  # Welch’s t-test
    stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
    y_max = max(max(g1), max(g2))
    
    # Median and CI95
    median1 = np.median(g1)
    ci1 = ci95(g1)
    median2 = np.median(g2)
    ci2 = ci95(g2)
    
    # Print values in console
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(f"{labels[i]}:", file=f)
        print(f"  Group 1: median={median1:.2f}, CI95={ci1}", file=f)
        print(f"  Group 2: median={median2:.2f}, CI95={ci2}", file=f)
        print(f"  p-value = {p_val:.4f}", file=f)
        print('----------------------------', file=f)
    
    # Annotate on plot
    x_pos = i * 3 + 1.5
    ax.text(x_pos, y_max + 1, f"p={p_val:.3f}", ha='center', fontsize=10)

# plt.tight_layout()
# plt.show()
print('compare gender')
for months_prior in labels:

    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(months_prior, file=f)
    with_rf_dict = Counter(with_and_without_rf[months_prior][2]['with_rf'])
    without_rf_dict = Counter(with_and_without_rf[months_prior][2]['without_rf'])
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(with_rf_dict, file=f)
        print(without_rf_dict, file=f)

    table = [[with_rf_dict['MALE'], with_rf_dict['FEMALE']], [without_rf_dict['MALE'], without_rf_dict['FEMALE']]]
    chi2, p, dof, expected = chi2_contingency(table)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(f"Chi-square p-value: {p:.4f}", file=f)
        print('----------------------------', file=f)
print('compare ethnicity')
for months_prior in labels:
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(months_prior, file=f)
    with_rf_dict = Counter(with_and_without_rf[months_prior][3]['with_rf'])
    without_rf_dict = Counter(with_and_without_rf[months_prior][3]['without_rf'])
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(with_rf_dict, file=f)
        print(without_rf_dict, file=f)

    # table = [[with_rf_dict['Unknown'], with_rf_dict['Not Hispanic or Latino'], with_rf_dict['Hispanic or Latino']], 
    #          [without_rf_dict['Unknown'], without_rf_dict['Not Hispanic or Latino'], without_rf_dict['Hispanic or Latino']]]
    # chi2, p, dof, expected = chi2_contingency(table)
    # with open('output/baseline_characteristics_log.txt', 'a') as f:
    #     print(f"Chi-square p-value: {p:.4f}", file=f)
    #     print('not hispanic vs hispanic', file=f)

    table = [[with_rf_dict['Not Hispanic or Latino'], with_rf_dict['Hispanic or Latino']], 
             [without_rf_dict['Not Hispanic or Latino'], without_rf_dict['Hispanic or Latino']]]
    chi2, p, dof, expected = chi2_contingency(table)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(f"Chi-square p-value: {p:.4f}", file=f)

print('compare race')
with_rf_race, without_rf_race = {}, {}
for months_prior in labels:
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(months_prior, file=f)
    with_rf_dict = Counter(with_and_without_rf[months_prior][4]['with_rf'])
    without_rf_dict = Counter(with_and_without_rf[months_prior][4]['without_rf'])
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(with_rf_dict, file=f)
        print(without_rf_dict, file=f)

    with_rf_dict = {
        'White': with_rf_dict.get('White', 0),
        'Black': with_rf_dict.get('Black or African American', 0),
        'Others': sum(v for k, v in with_rf_dict.items() if k not in ['White', 'Black or African American'])
    }

    without_rf_dict = {
        'White': without_rf_dict.get('White', 0),
        'Black': without_rf_dict.get('Black or African American', 0),
        'Others': sum(v for k, v in without_rf_dict.items() if k not in ['White', 'Black or African American'])
    }
    table = [[with_rf_dict['White'], with_rf_dict['Black'], with_rf_dict['Others']], 
             [without_rf_dict['White'], without_rf_dict['Black'], without_rf_dict['Others']]]
    chi2, p, dof, expected = chi2_contingency(table)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(f"Chi-square p-value: {p:.4f}", file=f)

    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print('white vs black', file=f)
    table = [[with_rf_dict['White'], with_rf_dict['Black']], 
             [without_rf_dict['White'], without_rf_dict['Black']]]
    chi2, p, dof, expected = chi2_contingency(table)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(f"Chi-square p-value: {p:.4f}", file=f)

    with_rf_race[months_prior] = with_rf_dict
    without_rf_race[months_prior] = without_rf_dict

median_age_with, median_age_without = {}, {}
for months_prior in labels:
    median_age_with[months_prior] = [np.median(i) for i in with_and_without_rf[months_prior][5]['with_rf']]
    median_age_without[months_prior] = [np.median(i) for i in with_and_without_rf[months_prior][5]['without_rf']]

# Example: two lists of age distributions for 3 time windows
group1, group2 = [], []
for months_prior in labels:
    group1.append(median_age_with[months_prior])
    group2.append(median_age_without[months_prior])

# Combine for grouped boxplot
data = []
positions = []
for i in range(3):
    data += [group1[i], group2[i]]
    positions += [i*3 + 1, i*3 + 2]  # space between groups

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(data, positions=positions, widths=0.6)

# Set x-axis
xtick_positions = [i*3 + 1.5 for i in range(3)]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(labels, fontsize = 12)
ax.set_ylabel('Age', fontsize = 12)
ax.legend([plt.Line2D([0], [0], color='black')], ['with RF vs without RF'], fontsize = 12)

print('compare median age of time window')
for i in range(3):
    g1 = group1[i]
    g2 = group2[i]
    
    # T-test
    # t_stat, p_val = ttest_ind(g1, g2, equal_var=False)  # Welch’s t-test
    stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
    y_max = max(max(g1), max(g2))
    
    # Median and CI95
    median1 = np.median(g1)
    ci1 = ci95(g1)
    median2 = np.median(g2)
    ci2 = ci95(g2)
    
    # Print values in console
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print('----------------------------', file=f)
        print(f"{labels[i]}:", file=f)
        print(f"  Group 1: median={median1:.2f}, CI95={ci1}", file=f)
        print(f"  Group 2: median={median2:.2f}, CI95={ci2}", file=f)
        print(f"  p-value = {p_val:.4f}", file=f)
    
    # Annotate on plot
    x_pos = i * 3 + 1.5
    ax.text(x_pos, y_max + 1, f"p={p_val:.3f}", ha='center', fontsize=10)

# plt.tight_layout()
# plt.show()
print('compare comorbidity prevalence')
for months_prior in labels:
    comorb_dict1 = Counter(with_and_without_rf[months_prior][0]['with_rf']).most_common(20)
    comorb_dict1 = {concept_dict[i[0]]:i[1] for i in comorb_dict1}
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(months_prior, file=f)
        print('with_rf', file=f)
        print(comorb_dict1, file=f)
        print('without_rf', file=f)
    comorb_dict2 = Counter(with_and_without_rf[months_prior][0]['without_rf']).most_common(20)
    comorb_dict2 = {concept_dict[i[0]]:i[1] for i in comorb_dict2}
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(comorb_dict2, file=f)
        print('----------------------------', file=f)
        print('common', file=f)
    common_comorb = set(list(comorb_dict1.keys())) & set(list(comorb_dict2.keys()))
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(common_comorb, file=f)
    for i in common_comorb:
        table = [[comorb_dict1[i], comorb_dict2[i]], 
                [comorb_dict1[i], comorb_dict2[i]]]
        chi2, p, dof, expected = chi2_contingency(table)
        if p<0.05:
            with open('output/baseline_characteristics_log.txt', 'a') as f:
                print(i, file=f)
                print(f"Chi-square p-value: {p:.4f}", file=f)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print('----------------------------', file=f)
        print('diff yes with rf no without rf', file=f)
    diff = list(set(list(comorb_dict1.keys()))-common_comorb)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(len(diff), diff, file=f)
        print('diff yes without rf no with rf', file=f)
    diff = list(set(list(comorb_dict2.keys()))-common_comorb)
    with open('output/baseline_characteristics_log.txt', 'a') as f:
        print(len(diff), diff, file=f)
        
hl = []
for k, v in pat_baseline.items():
    hl.append(len(v['concept_dx']))
with open('output/baseline_characteristics_log.txt', 'a') as f:
    print('median number of records per patient', file=f)
    print(np.median(hl), ci95(hl), file=f)