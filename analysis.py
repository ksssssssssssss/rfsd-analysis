"""
Репозиторий: https://github.com/ksssssssssssss/rfsd-analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

np.random.seed(42)


# 1.ГЕНЕРАЦИЯ ДАННЫХ
# Примечание: в реальном проекте данные загружаются с Hugging Face:
# df = pd.read_parquet("hf://datasets/irlspbru/RFSD/RFSD/year=2023/*.parquet")
# Здесь используется синтетический датасет, структурированный как RFSD.

n = 1000
log_size = np.random.normal(12, 2.5, n)
size = np.exp(log_size)

year = np.full(n, 2023)
B_assets = size * np.random.lognormal(0, 0.3, n)
B_current_assets = np.clip(B_assets * np.random.normal(0.55, 0.15, n), B_assets * 0.05, B_assets * 0.95)
B_equity = np.clip(B_assets * np.random.normal(0.42, 0.12, n), B_assets * 0.01, B_assets * 0.9)
B_inventory = np.clip(B_current_assets * np.random.normal(0.18, 0.08, n), 0, B_current_assets * 0.8)
B_long_term_liabilities = np.clip(B_assets * np.random.exponential(0.08, n), 0, B_assets * 0.5)
B_short_term_liabilities = np.clip(B_assets * np.random.exponential(0.12, n), 0, B_assets * 0.6)
B_cash = np.clip(B_current_assets * np.random.normal(0.10, 0.06, n), 0, B_current_assets * 0.5)
PL_revenue = B_assets * np.random.lognormal(0.4, 0.5, n)
PL_cost_of_sales = np.clip(PL_revenue * np.random.normal(0.78, 0.10, n), 0, PL_revenue * 0.98)
PL_profit_before_tax = PL_revenue - PL_cost_of_sales + np.random.normal(0, PL_revenue * 0.05, n)
PL_net_profit = PL_profit_before_tax * np.random.normal(0.78, 0.12, n)
PL_interest_payable = (B_long_term_liabilities + B_short_term_liabilities) * np.random.exponential(0.05, n)
PL_interest_receivable = B_cash * np.random.exponential(0.03, n)
CFo_materials = np.clip(PL_revenue * np.random.normal(0.42, 0.10, n), 0, PL_revenue * 0.8)
CFo_wages = np.clip(PL_revenue * np.random.normal(0.15, 0.05, n), 0, PL_revenue * 0.4)
CFo_taxes = np.clip(PL_revenue * np.random.normal(0.05, 0.02, n), 0, PL_revenue * 0.15)
age = np.clip(np.random.exponential(8, n) + np.random.normal(0, 2, n), 0, 50).astype(int)
okved_section = np.random.choice(['C','F','G','H','J','K','M','N','O','Q'], size=n, p=[0.15,0.08,0.20,0.10,0.12,0.08,0.10,0.07,0.05,0.05])
region_name = np.random.choice(['Москва','Санкт-Петербург','Татарстан','Свердловская обл.','Нижегородская обл.','Ростовская обл.','Челябинская обл.','Самарская обл.','Башкортостан','Краснодарский край'], size=n, p=[0.25,0.12,0.08,0.10,0.07,0.08,0.07,0.06,0.09,0.08])

# Добавляем выбросы (гиганты)
outlier_idx = np.random.choice(n, 5, replace=False)
for idx in outlier_idx:
    mult = np.random.uniform(50, 200)
    for col in ['B_assets','B_current_assets','B_equity','PL_revenue','PL_cost_of_sales','PL_profit_before_tax','PL_net_profit']:
        globals()[col][idx] *= mult

# Добавляем убыточные компании
loss_idx = np.random.choice(n, 30, replace=False)
for idx in loss_idx:
    PL_net_profit[idx] = -abs(PL_net_profit[idx]) * np.random.uniform(0.5, 2)
    PL_profit_before_tax[idx] = -abs(PL_profit_before_tax[idx]) * np.random.uniform(0.3, 1.5)

df = pd.DataFrame({
    'year': year, 'B_assets': B_assets.round(2), 'B_current_assets': B_current_assets.round(2),
    'B_equity': B_equity.round(2), 'B_inventory': B_inventory.round(2),
    'B_long_term_liabilities': B_long_term_liabilities.round(2),
    'B_short_term_liabilities': B_short_term_liabilities.round(2),
    'B_cash': B_cash.round(2), 'PL_revenue': PL_revenue.round(2),
    'PL_cost_of_sales': PL_cost_of_sales.round(2),
    'PL_profit_before_tax': PL_profit_before_tax.round(2),
    'PL_net_profit': PL_net_profit.round(2),
    'PL_interest_payable': PL_interest_payable.round(2),
    'PL_interest_receivable': PL_interest_receivable.round(2),
    'CFo_materials': CFo_materials.round(2), 'CFo_wages': CFo_wages.round(2),
    'CFo_taxes': CFo_taxes.round(2), 'age': age,
    'okved_section': okved_section, 'region_name': region_name
})

print(f"Датасет создан: {df.shape}")

# 2. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
print("\n" + "="*60)
print("ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("="*60)
desc = df.describe().T
desc['missing'] = df.isnull().sum()
desc['skewness'] = df.select_dtypes(include=[np.number]).skew()
desc['kurtosis'] = df.select_dtypes(include=[np.number]).kurtosis()
print(desc.round(2).to_string())
desc.to_csv('descriptive_statistics.csv')

# 3. ГИСТОГРАММЫ И ВЫБРОСЫ
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
hist_cols = ['B_assets', 'PL_revenue', 'PL_net_profit']

for idx, col in enumerate(hist_cols):
    data = df[col].dropna()
    log_data = np.log1p(data.clip(lower=0))
    sns.histplot(log_data, kde=True, ax=axes[idx,0], color='steelblue', bins=40)
    axes[idx,0].set_title(f'Гистограмма: {col} (log₁ₚ)')
    sns.boxplot(x=data, ax=axes[idx,1], color='coral')
    axes[idx,1].set_title(f'Boxplot: {col}')

    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    axes[idx,1].annotate(f'Выбросов: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)', 
                         xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('histograms_and_outliers.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. МАТРИЦА КОРРЕЛЯЦИИ
df_corr = df.copy()
df_corr['okved_section_enc'] = pd.Categorical(df_corr['okved_section']).codes
df_corr['region_name_enc'] = pd.Categorical(df_corr['region_name']).codes
df_corr = df_corr.drop(['okved_section', 'region_name'], axis=1)
df_corr = df_corr.rename(columns={'okved_section_enc': 'okved_section', 'region_name_enc': 'region_name'})
df_corr = df_corr.fillna(df_corr.median())

corr_matrix = df_corr.corr()

plt.figure(figsize=(18, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, linewidths=0.3, annot_kws={'size': 8})
plt.title('Матрица корреляции (20 колонок)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], r))

print("ВЫСОКАЯ КОРРЕЛЯЦИЯ (|r| > 0.8)")
for a, b, r in high_corr:
    print(f"  {a:30s} ↔ {b:30s}  r = {r:+.3f}")

# 5. SCATTER PLOT
x_col, y_col = 'B_assets', 'PL_revenue'
r_val = corr_matrix.loc[x_col, y_col]

plt.figure(figsize=(10, 8))
x_data = np.log1p(df_corr[x_col].clip(lower=0))
y_data = np.log1p(df_corr[y_col].clip(lower=0))
sns.scatterplot(x=x_data, y=y_data, alpha=0.6, s=60, color='darkgreen', edgecolor='white', linewidth=0.3)
plt.xlabel(f'log₁ₚ({x_col}) — Активы')
plt.ylabel(f'log₁ₚ({y_col}) — Выручка')
plt.title(f'Scatter plot: {x_col} vs {y_col} (Pearson r = {r_val:.3f})')

z = np.polyfit(x_data, y_data, 1)
p = np.poly1d(z)
plt.plot(np.sort(x_data), p(np.sort(x_data)), "r--", alpha=0.8, linewidth=2, label='Линия тренда')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. СОХРАНЕНИЕ
df.to_csv('rfsd_sample_20x1000.csv', index=False)
print("\nВсе файлы сохранены:")
print("   - descriptive_statistics.csv")
print("   - histograms_and_outliers.png")
print("   - correlation_matrix.png")
print("   - scatter_plot.png")
print("   - rfsd_sample_20x1000.csv")
