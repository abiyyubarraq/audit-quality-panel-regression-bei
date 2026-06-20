import pandas as pd
import numpy as np

# Load data
data = pd.read_json('../data/processed/audit_data.json')

print('\n' + '='*80)
print('DATA QUALITY CHECK')
print('='*80)
print(f'N observations: {len(data)}')
print(f'N entities: {data["CODE"].nunique()}')
print(f'Time periods: {sorted(data["YEAR"].unique())}')

print('\n' + '='*80)
print('ROA DISTRIBUTION (POTENTIAL ISSUE)')
print('='*80)
print(data['ROA'].describe())

print(f'\nROA outliers (|ROA| > 100):')
outliers = data[data['ROA'].abs() > 100]
print(f'Count: {len(outliers)} out of {len(data)} ({len(outliers)/len(data)*100:.1f}%)')
if len(outliers) > 0:
    print(outliers[['CODE', 'YEAR', 'ROA']].sort_values('ROA'))

print('\n' + '='*80)
print('WITHIN-ENTITY VARIATION ANALYSIS')
print('='*80)
print('(High ratio = mostly between-entity variation)')
print('(Low ratio = mostly within-entity variation over time)\n')

for var in ['ARL', 'FEE', 'FO', 'SIZE', 'ROA', 'AQMS']:
    within_std = data.groupby('CODE')[var].transform(lambda x: x - x.mean()).std()
    between_std = data.groupby('CODE')[var].mean().std()
    ratio = between_std / within_std if within_std > 0 else float('inf')
    print(f'{var:6s}: Between/Within ratio = {ratio:6.2f}  (Between SD={between_std:7.2f}, Within SD={within_std:7.2f})')

print('\n' + '='*80)
print('VARIABLES WITH LOW WITHIN VARIATION (Problematic for FE)')
print('='*80)
print('Fixed Effects only uses within-entity variation.')
print('If a variable has little within variation, FE cannot estimate its effect.\n')

for var in ['ARL', 'FEE', 'FO', 'SIZE']:
    within_std = data.groupby('CODE')[var].transform(lambda x: x - x.mean()).std()
    overall_std = data[var].std()
    pct_within = (within_std / overall_std) * 100 if overall_std > 0 else 0
    print(f'{var:6s}: Within variation = {pct_within:5.1f}% of total variation')

print('\n' + '='*80)
print('DEGREES OF FREEDOM CHECK')
print('='*80)
print(f'Total observations: {len(data)}')
print(f'Entity fixed effects: {data["CODE"].nunique()}')
print(f'Independent variables: 5 (ARL, FEE, FO, SIZE, ROA)')
print(f'Total parameters: {data["CODE"].nunique() + 5}')
print(f'Residual degrees of freedom: {len(data) - data["CODE"].nunique() - 5}')
print(f'\nRule of thumb: Need at least 10-15 observations per parameter')
print(f'Actual: {len(data) / (data["CODE"].nunique() + 5):.1f} observations per parameter')
