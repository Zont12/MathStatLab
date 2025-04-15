import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom, beta

table = pd.read_csv("iris.csv") # загружаем файл с таблицеЙ
table_size = table.size
vue = table.head()
print(vue)
species_count = table['Species'].value_counts()
print("Количество экземпляров каждого ириса")
print(species_count)
max = species_count.idxmax()
min = species_count.idxmin()
if max == min:
    print("Вида с наибольшим количеством или наименьшим количеством цветков не существует!")
    print("Вид, который представлен чаще всего:", max)
    print("Вид, который представлен реже всего:", min)
else:
    print("Вид, который представлен чаще всего:", max)
    print("Вид, который представлен реже всего:", min)

table['Sepal_square'] = table['Sepal.Length'] * table["Sepal.Width"]
table['Petal_square'] = table['Petal.Length'] * table["Petal.Width"]
table['Square'] = table['Sepal_square'] + table['Petal_square']
print(table['Square'] ) # суммарная площадь

print("Вывод рассчетов для всей площади")
sample_average = table['Square'].mean()
sample_dispersion = table['Square'].var()
sample_median = table['Square'].median()
sample_quantile = table['Square'].quantile(0.4)
print("Выборочное среднее:", sample_average)
print("Выборочная Дисперсия:", sample_dispersion)
print("Выборочная Медиана:", sample_median)
print("Выборочный Квантиль 0.4:", sample_quantile)


grouped = table.groupby("Species")["Square"]
for species, values in grouped:
    print(f"\nВид: {species}")
    print("Выборочное среднее:", values.mean())
    print("Выборочная Дисперсия:", values.var())
    print("Выборочная Медиана:", values.median())
    print("Выборочный Квантиль 0.4:", values.quantile(0.4))

def draw_efr(data, label='', xlabel='Значения', ylabel='Доля', title='Эмпирическая функция распределения'):
    sorted_table = np.sort(data)
    table_size = len(sorted_table)
    y = np.arange(1, table_size + 1) / table_size # доли значений
    plt.figure(figsize=(9, 6))
    plt.step(sorted_table, y, where="post", label=label if label else 'Доля', color='green', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True) # сетка на графике
    plt.legend() # подпись
    plt.show()

draw_efr(
    data=table['Square'],  # данные для построения
    label='Эмпирическая функция распределения',
    xlabel='Суммарная площадь',
    title='Эмпирическая функция распределения суммарной площали для всей совокупности '
)

for species in table['Species'].unique():
    species_data = table[table['Species'] == species]['Square']
    draw_efr(
        data=species_data,
        label='Эмпирическая функция распределения',
        xlabel=f'Суммарная площадь для {species} ',
        title=f'Эмпирическая функция распределения для {species} '
    )

def draw_histogram(data, x_column='Square', hue_column=None, title="Гистограмма", xlabel="Значение", ylabel="Частота"):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=x_column, hue=hue_column, bins=20, kde=False, palette='Set2')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

draw_histogram(
    data=table,
    x_column='Square',
    hue_column='Species',
    title='Гистограмма суммарной площади по видам',
    xlabel='Суммарная площадь',
    ylabel='Частота'
)


plt.figure(figsize=(6, 5))
plt.boxplot(table['Square'])
plt.title('Boxplot суммарной площади для всей совокупности)')
plt.ylabel('Суммарная площадь')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks([1], ['Все ирисы']) 
plt.show()


def draw_boxplot(dataframe, column='Square', group_column='Species', title='Boxplot по видам'):
    grouped_data = [group[column].values for _, group in dataframe.groupby(group_column)]
    labels = dataframe[group_column].unique()

    plt.figure(figsize=(8, 6))
    plt.boxplot(grouped_data, labels=labels, vert=True)
    plt.xlabel(group_column)
    plt.ylabel(column)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

table.rename(columns={'Square': 'Суммарная площадь'}, inplace=True)
table.rename(columns={'Species': 'Виды ирисов'}, inplace=True)
draw_boxplot(
    dataframe=table,
    column='Суммарная площадь',
    group_column='Виды ирисов',
    title='Boxplot суммарной площади по видам ирисов'
)

table.rename(columns={'Суммарная площадь': 'Square'}, inplace=True)
table.rename(columns={'Виды ирисов': 'Species'}, inplace=True)


################################################################################################## задание 2

# Смещённая дисперсия (MLE)
n = table.shape[0]


# Логнормальное распределение
log_square = np.log(table['Square'])
estimated_average_lognormal_distribution_in_log = log_square.mean()
estimated_variance_lognormal_distribution_in_log = ((log_square - estimated_average_lognormal_distribution_in_log) ** 2).mean()

# Переводим mu и Var в исходные единицы
estimated_average_lognormal_distribution = np.exp(estimated_average_lognormal_distribution_in_log + estimated_variance_lognormal_distribution_in_log / 2)
estimated_variance_lognormal_distribution = (
    np.exp(estimated_variance_lognormal_distribution_in_log) *
    np.exp(2 * estimated_average_lognormal_distribution_in_log + estimated_variance_lognormal_distribution_in_log)
    - np.exp(2 * estimated_average_lognormal_distribution_in_log + estimated_variance_lognormal_distribution_in_log)
)

print("\n--- Логнормальное распределение ---")
print("Оценка среднего:", estimated_average_lognormal_distribution)
print("Оценка дисперсии:", estimated_variance_lognormal_distribution)


# MSE
def mse_formula(sigma_squared, n):
    return (sigma_squared ** 2 / n ** 2) + (2 * sigma_squared ** 2 / n)

mse_lognormal = mse_formula(estimated_variance_lognormal_distribution, n)

print("\n--- MSE дисперсий ---")
print("MSE (логнормальное):", mse_lognormal)

# Bias дисперсий
bias_lognormal_distribution = estimated_variance_lognormal_distribution ** 2 / n

print("\n--- Смещения дисперсий (bias) ---")
print("Bias (логнормальное):", bias_lognormal_distribution)
# Информация Фишера для оценки среднего
info_fisher_mu_lognormal = n / estimated_variance_lognormal_distribution

print("\n--- Информация Фишера для μ ---")
print("Информация Фишера (логнормальное):", info_fisher_mu_lognormal)

# Информация Фишера для оценки дисперсии
info_fisher_sigma2_lognormal = n / (2 * estimated_variance_lognormal_distribution ** 2)

print("\n--- Информация Фишера для σ² ---")
print("Информация Фишера (логнормальное):", info_fisher_sigma2_lognormal)

############################################################################################ 3 задание 

# Заданные параметры
mu_0 = estimated_average_lognormal_distribution
sigma2_0 = estimated_variance_lognormal_distribution
sigma_0 = np.sqrt(sigma2_0)
n_values = [10, 50, 100, 500, 1000]
M = 100

mu_estimates = {n: [] for n in n_values}
sigma2_estimates = {n: [] for n in n_values}

for n in n_values:
    for _ in range(M):
        sample = np.random.normal(mu_0, sigma_0, size=n)
        mu_hat = np.mean(sample)
        sigma2_hat = np.var(sample, ddof=0)  # Смещённая оценка по MLE
        mu_estimates[n].append(mu_hat)
        sigma2_estimates[n].append(sigma2_hat)

# Формируем DataFrame'ы
df_mu = pd.DataFrame([(n, mu) for n in n_values for mu in mu_estimates[n]], columns=["n", "mu_hat"])
df_sigma2 = pd.DataFrame([(n, sigma2) for n in n_values for sigma2 in sigma2_estimates[n]], columns=["n", "sigma2_hat"])

# Визуализация
fig, axes = plt.subplots(3, 2, figsize=(12, 18), constrained_layout=True)
axes = axes.flatten()

sns.histplot(df_mu, x="mu_hat", hue="n", kde=True, bins=20, ax=axes[0])
axes[0].set_title("Гистограммы оценок μ")

sns.histplot(df_sigma2, x="sigma2_hat", hue="n", kde=True, bins=20, ax=axes[1])
axes[1].set_title("Гистограммы оценок σ²")

sns.boxplot(data=df_mu, x="n", y="mu_hat", ax=axes[2])
axes[2].set_title("Box-plot оценок μ")

sns.boxplot(data=df_sigma2, x="n", y="sigma2_hat", ax=axes[3])
axes[3].set_title("Box-plot оценок σ²")

sns.violinplot(data=df_mu, x="n", y="mu_hat", ax=axes[4], inner="quartile")
axes[4].set_title("Violin-plot оценок μ")

sns.violinplot(data=df_sigma2, x="n", y="sigma2_hat", ax=axes[5], inner="quartile")
axes[5].set_title("Violin-plot оценок σ²")

plt.show()
