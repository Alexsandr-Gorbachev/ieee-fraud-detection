"""
plots.py - Функции для визуализации данных и результатов

Содержит:
- Графики распределения признаков
- Анализ дисбаланса классов
- ROC-AUC кривые
- Feature importance визуализация
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif

# Настройки для красивых графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_class_distribution(y, title="Распределение классов"):
    """
    Показывает распределение классов в датасете (важно для понимания дисбаланса).
    
    Parameters:
    -----------
    y : pd.Series or array-like
        Таргет переменная
    title : str
        Заголовок графика
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Абсолютные числа
    counts = pd.Series(y).value_counts()
    axes[0].bar(counts.index, counts.values, color=['green', 'red'])
    axes[0].set_xlabel('Класс')
    axes[0].set_ylabel('Количество')
    axes[0].set_title(f'{title} (абсолютные числа)')
    axes[0].set_xticklabels(['Non-Fraud', 'Fraud'])
    
    # Проценты
    proportions = pd.Series(y).value_counts(normalize=True) * 100
    axes[1].bar(proportions.index, proportions.values, color=['green', 'red'])
    axes[1].set_xlabel('Класс')
    axes[1].set_ylabel('Процент (%)')
    axes[1].set_title(f'{title} (проценты)')
    axes[1].set_xticklabels(['Non-Fraud', 'Fraud'])
    
    for ax in axes:
        for i, v in enumerate(ax.get_height()):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nДисбаланс классов:")
    print(f"  Non-Fraud: {counts[0]} ({proportions[0]:.2f}%)")
    print(f"  Fraud:     {counts[1]} ({proportions[1]:.2f}%)")


def plot_missing_values(data, title="Пропущенные значения"):
    """
    Визуализирует пропущенные значения в датасете.
    """
    missing = data.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("Пропущенные значения не обнаружены")
        return
    
    missing_pct = (missing / len(data)) * 100
    
    # Динамически подбираем высоту графика
    height = max(6, len(missing) * 0.3)  # минимум 6, по 0.3 на признак
    
    fig, ax = plt.subplots(figsize=(12, height), dpi=120)  # DPI для четкости
    ax.barh(range(len(missing)), missing_pct.values)
    ax.set_yticks(range(len(missing)))
    ax.set_yticklabels(missing_pct.index, fontsize=9)  # уменьшили шрифт
    ax.set_xlabel('Процент пропущенных значений (%)', fontsize=11)
    ax.set_title(title, fontsize=13)
    
    # Подписи значений
    for i, v in enumerate(missing_pct.values):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    
    plt.tight_layout()  # убирает обрезание меток
    plt.show()


def analyze_transaction_amt(train, target_col='isFraud', figsize=(18, 12)):
    """
    Глубокий анализ TransactionAmt с 8 графиками и статистикой.
    
    Args:
        train: DataFrame с данными
        target_col: название целевой переменной
        figsize: размер фигуры
    """
    
    # Создаем большую фигуру с 8 графиками
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Глубокий анализ TransactionAmt', fontsize=16, fontweight='bold', y=0.995)
    
    # График 1: Распределение с логом
    ax1 = fig.add_subplot(gs[0, :2])
    for fraud_val in [0, 1]:
        subset = train[train[target_col] == fraud_val]['TransactionAmt']
        label = 'Fraud' if fraud_val == 1 else 'Normal'
        color = 'red' if fraud_val == 1 else 'green'
        ax1.hist(subset, bins=100, alpha=0.6, label=label, color=color, density=True)
    ax1.set_xscale('log')
    ax1.set_title('Распределение сумм: Fraud vs Normal', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Violin plot
    ax2 = fig.add_subplot(gs[0, 2])
    sns.violinplot(data=train, y='TransactionAmt', x=target_col, palette=['green', 'red'], ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Violin plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # График 3: Fraud Rate по квантилям
    ax3 = fig.add_subplot(gs[1, 0])
    train['amt_quantile'] = pd.qcut(train['TransactionAmt'], q=20, duplicates='drop')
    fraud_by_quantile = train.groupby('amt_quantile')[target_col].agg(['mean', 'count']).reset_index()
    fraud_by_quantile['fraud_rate'] = fraud_by_quantile['mean'] * 100
    x_pos = range(len(fraud_by_quantile))
    colors_bar = ['red' if x > train[target_col].mean()*100 else 'orange' for x in fraud_by_quantile['fraud_rate']]
    ax3.bar(x_pos, fraud_by_quantile['fraud_rate'], color=colors_bar, edgecolor='black', alpha=0.8)
    ax3.axhline(y=train[target_col].mean()*100, color='blue', linestyle='--', linewidth=2, label='Средний fraud rate')
    ax3.set_title('Fraud Rate по квантилям', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(0, len(fraud_by_quantile), 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # График 4: CDF
    ax4 = fig.add_subplot(gs[1, 1])
    for fraud_val in [0, 1]:
        subset = train[train[target_col] == fraud_val]['TransactionAmt'].sort_values()
        label = 'Fraud' if fraud_val == 1 else 'Normal'
        color = 'red' if fraud_val == 1 else 'green'
        ax4.plot(subset.values, np.linspace(0, 1, len(subset)), label=label, color=color, linewidth=2)
    ax4.set_xscale('log')
    ax4.set_title('CDF: сравнение распределений', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # График 5: Heatmap Сумма × Час
    ax5 = fig.add_subplot(gs[1, 2])
    train['amt_bin'] = pd.qcut(train['TransactionAmt'], q=10, duplicates='drop', labels=False)
    train['hour_of_day'] = (train['TransactionDT'] // 3600) % 24
    pivot = train.groupby(['amt_bin', 'hour_of_day'])[target_col].mean().reset_index().pivot(
        index='amt_bin', columns='hour_of_day', values=target_col) * 100
    sns.heatmap(pivot, cmap='RdYlGn_r', annot=False, cbar_kws={'label': 'Fraud %'}, ax=ax5)
    ax5.set_title('Fraud Rate: Сумма × Время', fontsize=12, fontweight='bold')
    
    # График 6: Boxplot
    ax6 = fig.add_subplot(gs[2, 0])
    train_sample = train[train['TransactionAmt'] < train['TransactionAmt'].quantile(0.95)]
    sns.boxplot(data=train_sample, x=target_col, y='TransactionAmt', palette=['green', 'red'], ax=ax6, showfliers=True)
    ax6.set_title('Boxplot без экстремальных выбросов', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # График 7: Топ-суммы
    ax7 = fig.add_subplot(gs[2, 1])
    top_amounts = train['TransactionAmt'].value_counts().head(15)
    colors_top = ['red' if train[train['TransactionAmt']==amt][target_col].mean() > 0.1 else 'steelblue' 
                  for amt in top_amounts.index]
    ax7.barh(range(len(top_amounts)), top_amounts.values, color=colors_top, edgecolor='black', alpha=0.8)
    ax7.set_yticks(range(len(top_amounts)))
    ax7.set_yticklabels([f'${x:.0f}' for x in top_amounts.index], fontsize=9)
    ax7.set_title('Топ-15 популярных сумм', fontsize=12, fontweight='bold')
    ax7.invert_yaxis()
    ax7.grid(True, alpha=0.3, axis='x')
    
    # График 8: Scatter
    ax8 = fig.add_subplot(gs[2, 2])
    sample = train.sample(min(10000, len(train)), random_state=42)
    scatter = ax8.scatter(sample['TransactionDT'] / (3600*24), sample['TransactionAmt'],
                         c=sample[target_col], cmap='RdYlGn_r', alpha=0.5, s=10)
    ax8.set_yscale('log')
    ax8.set_title('Scatter: Сумма × Время', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax8)
    
    plt.tight_layout()
    plt.show()
    
    # Текстовая статистика
    print("\n" + "="*70)
    print("СТАТИСТИКА TransactionAmt")
    print("="*70)
    
    for fraud_val in [0, 1]:
        label = "FRAUD" if fraud_val == 1 else "NORMAL"
        subset = train[train[target_col] == fraud_val]['TransactionAmt']
        print(f"\n{label}:")
        print(f"  Среднее: ${subset.mean():.2f}")
        print(f"  Медиана: ${subset.median():.2f}")
        print(f"  Std: ${subset.std():.2f}")
        print(f"  Min: ${subset.min():.2f}, Max: ${subset.max():.2f}")
        print(f"  Skewness: {subset.skew():.3f}")
    
    # Круглые суммы
    print("\n" + "-"*70)
    print("АНАЛИЗ 'КРУГЛЫХ' СУММ")
    print("-"*70)
    round_amounts = [50, 100, 200, 500, 1000, 2000, 5000]
    for amt in round_amounts:
        if amt in train['TransactionAmt'].values:
            count = (train['TransactionAmt'] == amt).sum()
            fraud_rate = train[train['TransactionAmt'] == amt][target_col].mean() * 100
            print(f"  ${amt}: {count:,} транзакций, fraud rate = {fraud_rate:.2f}%")
    
    print("\n" + "="*70)
    print("✓ Анализ завершен!")
    print("="*70)
    
    # Очистка временных колонок
    for col in ['amt_quantile', 'amt_bin', 'hour_of_day']:
        if col in train.columns:
            train.drop(columns=[col], inplace=True)
    
    return train


def analyze_transaction_dt(train, target_col='isFraud', figsize=(16, 10)):
    """
    Глубокий временной анализ TransactionDT с 4 графиками и статистикой.
    
    Args:
        train: DataFrame с данными
        target_col: название целевой переменной (опционально)
        figsize: размер фигуры
    """
    
    # Создаем временные признаки
    train['TransactionDT_hours'] = train['TransactionDT'] / 3600
    train['TransactionDT_days'] = train['TransactionDT'] / (3600 * 24)
    train['hour_of_day'] = (train['TransactionDT'] // 3600) % 24
    train['day_of_week'] = ((train['TransactionDT'] // (3600 * 24)) % 7).astype(int)
    
    # Создаем 4 графика
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Временной анализ TransactionDT', fontsize=16, fontweight='bold', y=1.00)
    
    # График 1: Распределение по времени (в днях)
    ax1 = axes[0, 0]
    ax1.hist(train['TransactionDT_days'], bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Распределение транзакций по времени', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    dt_min, dt_max = train['TransactionDT_days'].min(), train['TransactionDT_days'].max()
    dt_range = dt_max - dt_min
    ax1.text(0.02, 0.98, f'Период: {dt_range:.1f} дней\n{dt_min:.1f} - {dt_max:.1f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # График 2: По часам дня
    ax2 = axes[0, 1]
    hour_counts = train['hour_of_day'].value_counts().sort_index()
    colors = plt.cm.viridis(hour_counts.values / hour_counts.values.max())
    ax2.bar(hour_counts.index, hour_counts.values, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title('Распределение по часам дня', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3, axis='y')
    
    peak_hour = hour_counts.idxmax()
    min_hour = hour_counts.idxmin()
    ax2.axvline(x=peak_hour, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Пик: {peak_hour}ч')
    ax2.axvline(x=min_hour, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Мин: {min_hour}ч')
    ax2.legend()
    
    # График 3: По дням недели
    ax3 = axes[1, 0]
    day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    day_counts = train['day_of_week'].value_counts().sort_index()
    colors_day = ['#FF6B6B' if i >= 5 else '#4ECDC4' for i in range(7)]
    ax3.bar(range(7), [day_counts.get(i, 0) for i in range(7)], color=colors_day, edgecolor='black', alpha=0.8)
    ax3.set_title('Распределение по дням недели', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(day_names)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # График 4: Временной тренд
    ax4 = axes[1, 1]
    daily_counts = train.groupby(train['TransactionDT_days'].astype(int)).size()
    ax4.plot(daily_counts.index, daily_counts.values, color='darkblue', linewidth=1.5, alpha=0.7)
    ax4.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='steelblue')
    ax4.set_title('Временной тренд (по дням)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    if len(daily_counts) > 7:
        rolling_mean = daily_counts.rolling(window=7, center=True).mean()
        ax4.plot(rolling_mean.index, rolling_mean.values, color='red', linewidth=2, linestyle='--', label='7-дневное среднее')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print("\n" + "="*60)
    print("ВРЕМЕННАЯ СТАТИСТИКА TransactionDT")
    print("="*60)
    print(f"\nОбщий период: {dt_range:.1f} дней ({dt_range/7:.1f} недель)")
    print(f"Средних транзакций в день: {len(train) / dt_range:.0f}")
    print(f"Пиковый час: {peak_hour}:00 ({hour_counts[peak_hour]:,} транзакций)")
    print(f"Минимальный час: {min_hour}:00 ({hour_counts[min_hour]:,} транзакций)")
    print(f"Самый активный день: {day_names[day_counts.idxmax()]} ({day_counts.max():,})")
    print(f"Самый спокойный день: {day_names[day_counts.idxmin()]} ({day_counts.min():,})")
    
    # Fraud анализ
    if target_col in train.columns:
        print("\n" + "-"*60)
        print("FRAUD RATE ПО ВРЕМЕНИ")
        print("-"*60)
        
        hour_fraud = train.groupby('hour_of_day')[target_col].mean() * 100
        print(f"Самый рискованный час: {hour_fraud.idxmax()}:00 ({hour_fraud.max():.2f}% fraud)")
        print(f"Самый безопасный час: {hour_fraud.idxmin()}:00 ({hour_fraud.min():.2f}% fraud)")
        
        day_fraud = train.groupby('day_of_week')[target_col].mean() * 100
        print(f"Самый рискованный день: {day_names[day_fraud.idxmax()]} ({day_fraud.max():.2f}% fraud)")
        print(f"Самый безопасный день: {day_names[day_fraud.idxmin()]} ({day_fraud.min():.2f}% fraud)")
    
    # Очистка временных колонок (кроме hour_of_day, day_of_week - полезны)
    temp_cols = ['TransactionDT_hours', 'TransactionDT_days']
    train.drop(columns=[col for col in temp_cols if col in train.columns], inplace=True, errors='ignore')
    
    return train

def analyze_card_features_ultimate(train, card_features):
    """
ГРАФИК для  card_features
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Создаем данные
    risky_data = []
    for col in card_features[:6]:
        top_risky = train.groupby(col)['isFraud'].mean().sort_values(ascending=False).head(5)
        for val, rate in top_risky.items():
            if rate > 0.03:
                cnt_fraud = (train[(train[col] == val) & (train['isFraud'] == 1)]).sum()
                cnt_total = (train[col] == val).sum()
                risky_data.append([col, str(val), rate*100, cnt_fraud, cnt_total])
    
    df_risky = pd.DataFrame(risky_data, columns=['Карта', 'Значение', 'Fraud%', 'Фрод', 'Всего'])
    df_risky = df_risky.sort_values('Fraud%', ascending=True).head(20)
    
    # Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ЛЕВЫЙ: ТОП-20 рисков
    y_pos = range(len(df_risky))
    bars = ax1.barh(y_pos, df_risky['Fraud%'], color='red', alpha=0.8)
    ax1.set_xlabel('FRAUD RATE %', fontsize=14, fontweight='bold')
    ax1.set_title('ТОП-20 РИСКОВЫХ ЗНАЧЕНИЙ КАРТ', fontsize=16, fontweight='bold')
    
    # ✅ ИСПРАВЛЕНО: правильный доступ к колонке
    for i, (bar, fraud_pct) in enumerate(zip(bars, df_risky['Fraud%'])):
        ax1.text(bar.get_width() + 0.5, i, f'{fraud_pct:.1f}%', va='center', fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['Карта']}={row['Значение']}" for _, row in df_risky.iterrows()], fontsize=11)
    ax1.grid(axis='x', alpha=0.3)
    
    # ПРАВЫЙ: рисков по картам
    card_risk_counts = []
    for col in card_features[:6]:
        risky_count = len(train.groupby(col)['isFraud'].mean()[train.groupby(col)['isFraud'].mean() > 0.05])
        card_risk_counts.append(risky_count)
    
    colors = ['red' if count > 2 else 'orange' for count in card_risk_counts]
    ax2.bar(card_features[:6], card_risk_counts, color=colors, alpha=0.8)
    ax2.set_ylabel('КОЛИЧЕСТВО рисков', fontsize=14)
    ax2.set_title('РИСКОВЫЕ ЗНАЧЕНИЯ ПО КАРТАМ (>5%)', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Легенда
    fig.legend(['Транзакций', 'Fraud %'], loc='upper center', bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def analyze_addr_top_charts(train, addr_features):
    """
    ТОП ГРАФИКИ ДЛЯ addr_features из Kaggle IEEE CIS Fraud
    """
    
    
    # ✅ 2 Heatmap addr1 x addr2 (оставляем)
    plt.figure(figsize=(12, 10))
    top_addr1_idx = train['addr1'].value_counts().head(10).index
    top_addr2_idx = train['addr2'].value_counts().head(10).index
    
    sub_train = train[train['addr1'].isin(top_addr1_idx) & 
                     train['addr2'].isin(top_addr2_idx)]
    pivot = sub_train.groupby(['addr1', 'addr2'])['isFraud'].mean().unstack(fill_value=0) * 100
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Fraud %'})
    plt.title('addr1 × addr2: РИСКОВЫЕ КОМБИНАЦИИ\n(КРАСНЫЙ=опасно, ЗЕЛЕНЫЙ=безопасно)', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    
    print("\n📊 ГРАФИК (ТЕПЛОВАЯ КАРТА):")
    print("  • Строки = addr1 (регионы)")
    print("  • Столбцы = addr2 (страны)") 
    print("  • Цифры = % мошенничества")
    print("  • КРАСНЫЙ = ОПАСНО (>5%)")
    print("  • ЗЕЛЕНЫЙ = БЕЗОПАСНО (<1%)")
    
    # ✅ КЛЮЧЕВЫЕ ВЫВОДЫ
    print("\n" + "🚨 РЕЗУЛЬТАТЫ АНАЛИЗА" + "="*50)
    print(f"✅ Норма фрода: {train['isFraud'].mean()*100:.1f}%")

    
def analyze_c_features_simple(train, c_features):
    """
    ПРОСТОЙ АНАЛИЗ C_features - С ПОЛНЫМИ ПОЯСНЕНИЯМИ!
    """
    
    # 1. ТОП-10 C_features по разнице ФРОД vs НОРМАЛЬНЫЕ
    plt.figure(figsize=(12, 8))
    
    diff_stats = []
    for col in c_features[:20]:
        fraud_mean = train[train['isFraud']==1][col].mean()
        normal_mean = train[train['isFraud']==0][col].mean()
        diff = normal_mean - fraud_mean
        diff_stats.append((col, diff))
    
    diff_stats.sort(key=lambda x: x[1], reverse=True)
    top10 = diff_stats[:10]
    
    plt.barh(range(10), [x[1] for x in top10], color='green', alpha=0.7)
    plt.yticks(range(10), [x[0] for x in top10])
    plt.xlabel('НОРМАЛЬНЫЕ - ФРОД (больше = лучше фича)')
    plt.title('ТОП-10 C_features\n(длиннее бар = лучше фича)', fontsize=16)
    
    for i, (col, diff) in enumerate(top10):
        plt.text(diff + 0.01, i, f'{diff:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 2. HISTOGRAMM ТОП-1 фичи
    top_feature = top10[0][0]
    plt.figure(figsize=(12, 6))
    
    plt.hist(train[train['isFraud']==0][top_feature], bins=50, alpha=0.7, label='Нормальные', color='blue')
    plt.hist(train[train['isFraud']==1][top_feature], bins=50, alpha=0.7, label='ФРОД', color='red')
    
    plt.xlabel(f'{top_feature}')
    plt.ylabel('КОЛИЧЕСТВО ЛЮДЕЙ')
    plt.title(f'{top_feature}: ФРОД СЛЕВА, НОРМА СПРАВА!', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ✅ С ПОЛНЫМИ ПОЯСНЕНИЯМИ!
    print("\n" + "="*70)
    print("📖 ЧТО ПОКАЗАЛИ ГРАФИКИ:")
    print("="*70)
    print("\n📊 ПЕРВЫЙ ГРАФИК (горизонтальные зеленые бары):")
    print("  • ИМЯ СЛЕВА = название фичи (C1, C2, C13...)")
    print("  • ДЛИНА БАРА = насколько нормальные активнее фрода")
    print("  • ЧИСЛО СПРАВА = разница (0.123 = нормальные на 0.123 активнее)")
    print("  • ВЕРХ = ЛУЧШИЕ фичи для модели!")
    
    print("\n📊 ВТОРОЙ ГРАФИК (гистограмма):")
    print("  • СИНИЙ ХВОСТ СПРАВА = нормальные клиенты (много покупок)")
    print("  • КРАСНЫЙ ХВОСТ СЛЕВА = фрод (мало покупок)")
    print("  • X ось = количество покупок за N секунд")
    print("  • Y ось = сколько людей с таким количеством")
    
    print("\n🔥 ЗНАЧЕНИЯ:")
    print(f"ТОП фича: {top_feature}")
    print(f"Нормальные: {train[train['isFraud']==0][top_feature].mean():.1f} покупок")
    print(f"ФРОД:       {train[train['isFraud']==1][top_feature].mean():.1f} покупок")
    
    print("\n💡 ФИЧА ДЛЯ МОДЕЛИ:")
    print(f"train['низкая_активность'] = (train['{top_feature}'] < 1)")
    
def analyze_d_features_simple(train, d_features):
    """
    ПРОСТЫЕ ГРАФИКИ ДЛЯ D_features - ПОЛНОЕ ПОНЯТИЕ!
    D1-D15 = дней с последней транзакции
    """
    
    print("🔥 D_features = СКОЛЬКО ДНЕЙ НАЗАД БЫЛА ПОСЛЕДНЯЯ ПОКУПКА")
    print("D1 = дней назад по этой карте")
    print("D2 = дней назад по этому email")
    
    # 1. САМЫЙ ПРОСТОЙ ГРАФИК - D1 фрод vs нормальные
    plt.figure(figsize=(12, 8))
    
    d1_normal = train[train['isFraud']==0]['D1'].dropna()
    d1_fraud = train[train['isFraud']==1]['D1'].dropna()
    
    plt.hist(d1_normal, bins=50, alpha=0.7, label=f'Нормальные\nсреднее={d1_normal.mean():.1f} дней', color='blue')
    plt.hist(d1_fraud, bins=50, alpha=0.7, label=f'ФРОД\nсреднее={d1_fraud.mean():.1f} дней', color='red')
    
    plt.xlabel('D1 (дней с последней покупки)')
    plt.ylabel('КОЛИЧЕСТВО ЛЮДЕЙ')
    plt.title('D1: ФРОД ДАВНО НЕ ПОКУПАЛ!', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ ГРАФИК 1 - ЧТО ЗНАЧИТ:")
    print("• СИНИЙ график = нормальные клиенты")
    print("• КРАСНЫЙ график = фрод")
    print("• ФРОД ПРАВЕЕ = дольше не покупали!")
    
    # 2. ТОП-5 D_features по разнице
    plt.figure(figsize=(12, 8))
    
    diff_stats = []
    for col in ['D1','D2','D3','D4','D5']:
        if col in train.columns:
            fraud_mean = train[train['isFraud']==1][col].mean()
            normal_mean = train[train['isFraud']==0][col].mean()
            diff = fraud_mean - normal_mean
            diff_stats.append((col, diff))
    
    diff_stats.sort(key=lambda x: x[1], reverse=True)
    top5 = diff_stats[:5]
    
    plt.barh(range(5), [x[1] for x in top5], color='orange', alpha=0.8)
    plt.yticks(range(5), [x[0] for x in top5])
    plt.xlabel('ФРОД - НОРМАЛЬНЫЕ (дней)')
    plt.title('ТОП-5 D_features\n(правее = фрод дольше не покупал)', fontsize=16)
    
    for i, (col, diff) in enumerate(top5):
        plt.text(diff + 0.1, i, f'{diff:.1f} дней', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ ГРАФИК 2 - ЧТО ЗНАЧИТ:")
    print("• ОРАНЖЕВЫЕ БАРЫ = фрод дольше не покупал")
    print("• ДЛИНА БАРА = разница в днях")
    print("• D1 самый длинный = ЛУЧШАЯ ФИЧА!")
    
    # ЧИСЛА
    print("\n🔥 КОНКРЕТНЫЕ ЧИСЛА:")
    for col, diff in top5:
        fraud_mean = train[train['isFraud']==1][col].mean()
        normal_mean = train[train['isFraud']==0][col].mean()
        print(f"{col}: нормальные={normal_mean:.1f}д, фрод={fraud_mean:.1f}д, разница={diff:.1f}д")
    
    print("\n💡 ФИЧИ ДЛЯ МОДЕЛИ:")
    print("train['d1_старый'] = (train['D1'] > 30)")
    print("train['d1_d2_разница'] = train['D1'] - train['D2']")
    
    
def analyze_v_features_top(train, v_features):
    """
    V_features с ПОЛНЫМИ ПОЯСНЕНИЯМИ ПОД КАЖДЫМ ГРАФИКОМ!
    """
    
    print("🔥 V_features = 338 СЕКРЕТНЫХ фичей от банка (PCA!)")
    
    # 1. PCA 2D ПРОЕКЦИЯ
    print("\n📊 ГРАФИК 1: PCA - фрод отделен?")
    v_sample = v_features[:10]
    v_data = train[v_sample].fillna(0)
    
    scaler = StandardScaler()
    v_scaled = scaler.fit_transform(v_data)
    pca = PCA(n_components=2)
    v_pca = pca.fit_transform(v_scaled)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(v_pca[:,0], v_pca[:,1], 
                         c=train['isFraud'], cmap='RdYlBu_r', alpha=0.6, s=30)
    
    plt.xlabel(f'PC1 = {pca.explained_variance_ratio_[0]:.1%} информации')
    plt.ylabel(f'PC2 = {pca.explained_variance_ratio_[1]:.1%} информации')
    plt.title('V_features PCA: КРАСНЫЕ=ФРОД, СИНИЕ=НОРМАЛЬНЫЕ', fontsize=16)
    plt.colorbar(scatter, label='0=норм, 1=фрод')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("✅ ВЫВОД ГРАФИКА 1:")
    print("• КРАСНЫЕ точки = ФРОД")
    print("• СИНИЕ точки = нормальные транзакции")
    print(f"• Объясняет {sum(pca.explained_variance_ratio_):.1%} информации")
    print("• Если красные отдельно = V_features отличные!")
    
    # 2. ТОП-10 V фичей по корреляции
    print("\n📊 ГРАФИК 2: Какие V фичи лучше ловят фрод?")
    plt.figure(figsize=(12, 8))
    
    v_corr = train[v_features[:50] + ['isFraud']].corr()['isFraud'].sort_values(ascending=False)
    top_v = v_corr[1:11]
    
    plt.barh(range(10), top_v.values, color='purple', alpha=0.7)
    plt.yticks(range(10), top_v.index)
    plt.xlabel('СИЛА СВЯЗИ С ФРОДОМ (ближе к 1 = лучше)')
    plt.title('ТОП-10 V_features по силе предсказания фрода', fontsize=16)
    
    for i, corr in enumerate(top_v.values):
        plt.text(corr + 0.0005, i, f'{corr:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ ВЫВОД ГРАФИКА 2:")
    print("• ДЛИНА ФИОЛЕТОВОГО БАРА = сила предсказания")
    print("• Правее = лучше фича для модели")
    print(f"• ЛУЧШАЯ: {top_v.index[0]} (корреляция {top_v.values[0]:.4f})")
    
    # 3. РАСПРЕДЕЛЕНИЕ ТОП V фичи
    print("\n📊 ГРАФИК 3: Как выглядит топ фича?")
    top_v_feature = top_v.index[0]
    plt.figure(figsize=(14, 8))
    
    plt.hist(train[train['isFraud']==0][top_v_feature].dropna(), bins=50, alpha=0.7, 
             label=f'НОРМАЛЬНЫЕ (среднее={train[train["isFraud"]==0][top_v_feature].mean():.2f})', 
             color='blue', density=True)
    plt.hist(train[train['isFraud']==1][top_v_feature].dropna(), bins=50, alpha=0.7, 
             label=f'ФРОД (среднее={train[train["isFraud"]==1][top_v_feature].mean():.2f})', 
             color='red', density=True)
    
    plt.xlabel(f'{top_v_feature} (значение фичи)')
    plt.ylabel('ПЛОТНОСТЬ (сколько % транзакций)')
    plt.title(f'{top_v_feature}: ФРОД и НОРМАЛЬНЫЕ в РАЗНЫХ местах!', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("✅ ВЫВОД ГРАФИКА 3:")
    print(f"• СИНИЙ ХВОСТ = нормальные значения {top_v_feature}")
    print(f"• КРАСНЫЙ ХВОСТ = фрод значения {top_v_feature}")
    print("• Если хвосты не пересекаются = отличная фича!")
    
    # ИТОГОВЫЕ ВЫВОДЫ
    print("\n" + "="*70)
    print("🎯 ИТОГОВЫЕ ВЫВОДЫ ПО V_features:")
    print("="*70)
    print(f"1. PCA: фрод {sum(train['isFraud'])*100/len(train):.2f}% отделен")
    print(f"2. ТОП фича: {top_v_feature}")
    print(f"3. Всего фичей: {len(v_features)}")
    
    print("\n💎 ДЛЯ МОДЕЛИ БЕРИ:")
    print(f"v_top10 = ['{top_v_feature}', ...]  # топ-10 из графика 2")
    

def plot_correlation_matrix(data, columns=None, figsize=(14, 12)):
    """
    Корреляционная матрица между признаками (помогает найти мультиколлинеарность).
    
    Parameters:
    -----------
    data : pd.DataFrame
    columns : list, optional
        Какие колонки использовать (если None, все числовые)
    figsize : tuple
        Размер графика
    """
    if columns is None:
        # Берём числовые колонки
        data_numeric = data.select_dtypes(include=[np.number])
        if data_numeric.shape[1] > 50:
            # Если слишком много, выбираем первые 50
            data_numeric = data_numeric.iloc[:, :50]
    else:
        data_numeric = data[columns]
    
    corr = data_numeric.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Корреляционная матрица')
    plt.tight_layout()
    plt.show()
    
    
    
def plot_id_features_fraud_lift(
    train,
    target='isFraud',
    prefix='id',
    exclude=('TransactionID',),
    min_group_size=50,
    top_k_per_feat=3,
    lift_thr=2.0,
    top_n_plot=15,
    figsize=(16, 6),
    print_top=5
):
    # id-фичи
    id_feats = [c for c in train.columns if c.startswith(prefix) and c not in exclude]
    baseline = train[target].mean()

    # 1) Собираем топ категорий с большим lift
    all_top = []
    for feat in id_feats:
        if train[feat].dtype != 'object':
            continue

        fraud_lift = train.groupby(feat)[target].apply(
            lambda x: (x.mean() / baseline) if len(x) > min_group_size else 0
        )
        top_cat = fraud_lift.nlargest(top_k_per_feat)

        for cat, lift in top_cat.items():
            if lift > lift_thr:
                all_top.append((feat, cat, float(lift)))

    top_df = pd.DataFrame(all_top, columns=['id_feat', 'value', 'fraud_lift'])
    if len(top_df):
        top_df = top_df.sort_values('fraud_lift', ascending=False).head(top_n_plot)

    # 2) Missing rate
    missing_rate = train[id_feats].isnull().mean() * 100

    # --- Плотим ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if len(top_df):
        # чтобы было видно, какая категория: склеим в label
        plot_df = top_df.copy()
        plot_df['label'] = plot_df['id_feat'].astype(str) + ' = ' + plot_df['value'].astype(str)
        plot_df.sort_values('fraud_lift', ascending=True).plot(
            kind='barh', x='label', y='fraud_lift', ax=ax1, color='red', legend=False
        )
        ax1.axvline(lift_thr, color='green', ls='--')
    else:
        ax1.text(0.5, 0.5, f'Нет категорий с lift > {lift_thr}', ha='center', va='center')
        ax1.set_axis_off()

    ax1.set_title(f'ТОП Fraud ID категории (lift > {lift_thr}x)')
    ax1.set_xlabel('Lift vs baseline')

    missing_rate.sort_values(ascending=False).plot(kind='bar', ax=ax2, color='orange')
    ax2.set_title(f'Missing % по {prefix}_features')
    ax2.set_ylabel('Missing %')

    plt.tight_layout()
    plt.show()

    if print_top and len(top_df):
        print("🔥 ТОП KILLERS:")
        print(top_df.head(print_top))

    return top_df, missing_rate, id_feats
    
    
def weekly_fraud_analysis(train):
    """Недельная агрегация"""
    df = train.copy()
    df["dt_day"]  = ((df["TransactionDT"] - df["TransactionDT"].min()) // 86400).astype(int)
    df["dt_week"] = (df["dt_day"] // 7).astype(int)
    
    weekly = (
        df.groupby("dt_week")
          .agg(
              n=("TransactionID", "size"),
              fraud_n=("isFraud", "sum"),
              fraud_rate=("isFraud", "mean"),
              amt_median=("TransactionAmt", "median"),
              amt_mean=("TransactionAmt", "mean"),
          )
          .reset_index()
    )
    return weekly


def plot_weekly_fraud(weekly):
    """График недельный"""
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(weekly["dt_week"], weekly["fraud_rate"], marker="o")
    ax1.set_xlabel("Неделя от начала", fontsize=12)
    ax1.set_ylabel("Доля фрода", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.bar(weekly["dt_week"], weekly["n"], alpha=0.2)
    ax2.set_ylabel("Количество транзакций", fontsize=12)
    
    plt.title("Недельная доля фрода и объём транзакций", fontsize=13)
    plt.tight_layout()
    plt.show()


def daily_fraud_analysis(train):
    """Дневная агрегация"""
    df = train.copy()
    df["dt_day"] = ((df["TransactionDT"] - df["TransactionDT"].min()) // 86400).astype(int)
    
    daily = (
        df.groupby("dt_day")
          .agg(
              n=("TransactionID", "size"),
              fraud_n=("isFraud", "sum"),
              fraud_rate=("isFraud", "mean"),
          )
          .reset_index()
          .sort_values("dt_day")
    )
    daily["fraud_rate_ma7"] = daily["fraud_rate"].rolling(7, min_periods=1).mean()
    return daily


def plot_daily_fraud(daily):
    """График дневной"""
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(daily["dt_day"], daily["fraud_rate"], alpha=0.35, linewidth=1, label="Доля фрода (дневная)")
    ax1.plot(daily["dt_day"], daily["fraud_rate_ma7"], linewidth=2, label="Доля фрода (MA7)")
    ax1.set_xlabel("День от начала", fontsize=14)
    ax1.set_ylabel("Доля фрода", fontsize=14)
    ax1.grid(True, alpha=0.25)
    
    ax2 = ax1.twinx()
    ax2.bar(daily["dt_day"], daily["n"], alpha=0.15, width=1.0, label="Количество транзакций")
    ax2.set_ylabel("Количество транзакций", fontsize=14)
    
    ax1.set_title("Дневная доля фрода и объём транзакций (со сглаживанием 7 дней)", fontsize=17)
    ax1.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.show()
    
    

def analyze_missing_by_groups(train, feature_groups):
    """
    Анализ пропусков по группам признаков с визуализацией
    
    Parameters:
    -----------
    train : DataFrame
        Исходный датафрейм
    feature_groups : dict
        Словарь с группами признаков {group_name: [col1, col2, ...]}
        
    Returns:
    --------
    DataFrame : summary_df с статистикой по группам
    """
    
    # Создаем правильную сводку
    summary_data = []

    for group_name, cols in sorted(feature_groups.items()):
        cols_in_df = [c for c in cols if c in train.columns]
        if not cols_in_df:
            continue
        
        sub = train[cols_in_df]
        missing_count = sub.isna().sum().sum()
        total_cells = len(train) * len(cols_in_df)
        affected = (sub.isna().sum() > 0).sum()
        
        summary_data.append({
            'Group': group_name,
            'Count': len(cols_in_df),
            'Missing_%': round(missing_count / total_cells * 100, 2),
            'Affected': affected,
        })

    summary_df = pd.DataFrame(summary_data).sort_values('Missing_%', ascending=False)

    print("ИТОГОВАЯ СВОДКА ПО ГРУППАМ:")
    print(summary_df)
    print("\n")

    # Улучшенная визуализация с цветовой кодировкой
    fig, ax = plt.subplots(figsize=(12, 8))

    # Цвета в зависимости от уровня пропусков
    def get_color(missing_pct):
        if missing_pct > 60:
            return '#d62728'  # красный - критично
        elif missing_pct > 40:
            return '#ff7f0e'  # оранжевый - высоко
        elif missing_pct > 10:
            return '#ffbb78'  # светло-оранжевый - средне
        else:
            return '#2ca02c'  # зеленый - хорошо

    colors = [get_color(x) for x in summary_df['Missing_%']]

    # Scatter с размером пропорциональным количеству признаков
    scatter = ax.scatter(
        summary_df['Missing_%'], 
        summary_df['Count'],
        s=summary_df['Count'] * 10,  # размер пропорционален количеству
        c=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=2
    )

    # Аннотации
    for idx, row in summary_df.iterrows():
        ax.annotate(
            f"{row['Group']}\n({row['Affected']}/{row['Count']})",
            (row['Missing_%'], row['Count']),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
        )

    # Зоны по критичности
    ax.axvspan(0, 10, alpha=0.1, color='green', label='Низкий уровень пропусков')
    ax.axvspan(10, 40, alpha=0.1, color='yellow', label='Средний уровень')
    ax.axvspan(40, 60, alpha=0.1, color='orange', label='Высокий уровень')
    ax.axvspan(60, 100, alpha=0.1, color='red', label='Критический уровень')

    ax.set_xlabel('Доля пропусков (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество признаков в группе', fontsize=12, fontweight='bold')
    ax.set_title('Матрица качества данных по группам признаков', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)

    # Установим разумные лимиты
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, max(summary_df['Count']) * 1.15)

    plt.tight_layout()
    plt.show()

    # Дополнительная статистика
    print("\n" + "="*70)
    print("ПРИОРИТИЗАЦИЯ ГРУПП ДЛЯ ОБРАБОТКИ")
    print("="*70)

    for idx, row in summary_df.iterrows():
        group = row['Group']
        missing = row['Missing_%']
        count = row['Count']
        affected = row['Affected']
        
        if missing > 60:
            priority = "🔴 КРИТИЧНО"
            action = "Рассмотреть дроп или специальную обработку"
        elif missing > 40:
            priority = "🟠 ВЫСОКИЙ"
            action = "Создать is_missing флаги + заполнение"
        elif missing > 10:
            priority = "🟡 СРЕДНИЙ"
            action = "Простое заполнение (median/mode)"
        else:
            priority = "🟢 НИЗКИЙ"
            action = "Минимальная обработка"
        
        print(f"\n{group}:")
        print(f"  Приоритет: {priority}")
        print(f"  Пропуски: {missing}% ({affected}/{count} признаков затронуты)")
        print(f"  Действие: {action}")
    
    return summary_df 



def print_confusion_matrix_analysis(cm, y_true, y_pred, fold_name="ПОСЛЕДНИЙ ФОЛД"):
    """
    Выводит детальный анализ confusion matrix с метриками
    
    Parameters:
    -----------
    cm : array
        Confusion matrix от sklearn.metrics.confusion_matrix
    y_true : array
        Истинные метки
    y_pred : array
        Предсказанные метки
    fold_name : str
        Название фолда для отображения
    """
    print("\n" + "="*80)
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ CONFUSION MATRIX ({fold_name})")
    print("="*80)
    
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print(f"\n                  Predicted")
    print(f"              No Fraud  |  Fraud")
    print(f"         ─────────────────────────")
    print(f"Actual   |              |        ")
    print(f"No Fraud |    {tn:6d}  |  {fp:6d}")
    print(f"  Fraud  |    {fn:6d}  |  {tp:6d}")
    
    print(f"\n╔════════════════════════════════════════════╗")
    print(f"║  True Negatives  (TN): {tn:6d} ({tn/total*100:5.2f}%)  ║")
    print(f"║  False Positives (FP): {fp:6d} ({fp/total*100:5.2f}%)  ║")
    print(f"║  False Negatives (FN): {fn:6d} ({fn/total*100:5.2f}%)  ║")
    print(f"║  True Positives  (TP): {tp:6d} ({tp/total*100:5.2f}%)  ║")
    print(f"╚════════════════════════════════════════════╝")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\nДополнительные метрики из CM:")
    print(f"  Sensitivity (TPR, Recall): {sensitivity:.4f}  # доля найденного фрода")
    print(f"  Specificity (TNR):         {specificity:.4f}  # доля правильно распознанных не-фродов")
    print(f"  False Positive Rate (FPR): {fpr_rate:.4f}     # доля ложных тревог")
    print(f"  False Negative Rate (FNR): {fnr_rate:.4f}     # доля пропущенного фрода")
    
    # Classification Report
    print("\n" + "="*80)
    print(f"CLASSIFICATION REPORT ({fold_name})")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=['No Fraud', 'Fraud']))
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'sensitivity': sensitivity, 'specificity': specificity,
        'fpr': fpr_rate, 'fnr': fnr_rate
    }


def print_model_metrics(y_true, y_pred_proba, y_pred, fold_name="ПОСЛЕДНИЙ ФОЛД"):
    """
    Выводит основные метрики модели
    
    Parameters:
    -----------
    y_true : array
        Истинные метки
    y_pred_proba : array
        Вероятности предсказаний
    y_pred : array
        Предсказанные метки (бинарные)
    fold_name : str
        Название фолда
    """
    print("\n" + "="*100)
    print(f"МЕТРИКИ НА ТЕСТОВОЙ ЧАСТИ {fold_name}")
    print("="*100)
    
    test_auc = roc_auc_score(y_true, y_pred_proba)
    test_acc = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)
    
    print(f"\nROC-AUC:   {test_auc:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    
    return {
        'auc': test_auc,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }


def plot_model_evaluation(y_true, y_pred_proba, y_pred, fold_name="Last Fold", 
                          threshold=0.5, figsize=(14, 10)):
    """
    Строит 4 графика: ROC, PR Curve, Confusion Matrix, Distribution
    
    Parameters:
    -----------
    y_true : array
        Истинные метки
    y_pred_proba : array
        Вероятности предсказаний
    y_pred : array
        Предсказанные метки (бинарные)
    fold_name : str
        Название фолда для графиков
    threshold : float
        Порог классификации
    figsize : tuple
        Размер фигуры
    """
    print("\n" + "="*100)
    print(f"ПОСТРОЕНИЕ ГРАФИКОВ ({fold_name})")
    print("="*100)
    
    # Вычисляем метрики для графиков
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    test_auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"ROC-AUC: {test_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ===== График 1: ROC Curve =====
    axes[0, 0].plot(fpr, tpr, color='steelblue', linewidth=2, 
                    label=f'ROC curve (AUC = {test_auc:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1, 
                    label='Random classifier')
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 0].set_title(f'ROC Curve ({fold_name})', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(alpha=0.3)
    
    # ===== График 2: Precision-Recall Curve =====
    axes[0, 1].plot(recall_curve, precision_curve, color='coral', linewidth=2, 
                    label=f'PR curve (AUC = {pr_auc:.4f})')
    no_skill = y_true.sum() / len(y_true)
    axes[0, 1].plot([0, 1], [no_skill, no_skill], color='red', linestyle='--', 
                    linewidth=1, label=f'Baseline ({no_skill:.3f})')
    axes[0, 1].set_xlabel('Recall', fontsize=11)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title(f'Precision-Recall Curve ({fold_name})', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(alpha=0.3)
    
    # ===== График 3: Confusion Matrix =====
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Fraud', 'Fraud'],
        yticklabels=['No Fraud', 'Fraud'],
        cbar=True, square=True,
        linewidths=2, linecolor='black',
        annot_kws={"size": 16, "weight": "bold"},
        ax=axes[1, 0]
    )
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f'Confusion Matrix ({fold_name})', fontsize=13, fontweight='bold')
    
    # ===== График 4: Распределение вероятностей =====
    axes[1, 1].hist(
        y_pred_proba[y_true == 0], bins=50, alpha=0.6, color='steelblue',
        label='No Fraud', edgecolor='black', linewidth=0.5
    )
    axes[1, 1].hist(
        y_pred_proba[y_true == 1], bins=50, alpha=0.6, color='coral',
        label='Fraud', edgecolor='black', linewidth=0.5
    )
    axes[1, 1].axvline(x=threshold, color='green' if threshold != 0.5 else 'red', 
                       linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
    if threshold != 0.5:
        axes[1, 1].axvline(x=0.5, color='red', linestyle=':', linewidth=1.5, 
                           alpha=0.5, label='Default = 0.5')
    axes[1, 1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title(f'Distribution of Predicted Probabilities ({fold_name})', 
                         fontsize=13, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {'auc': test_auc, 'pr_auc': pr_auc, 'cm': cm}


def plot_feature_importance(importance_df, categorical_features, X_columns, 
                            title_prefix="", figsize=(16, 6)):
    """
    Строит графики важности признаков и кумулятивного вклада
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame с колонками 'feature' и 'importance'
    categorical_features : list
        Список категориальных признаков
    X_columns : Index
        Названия всех колонок X
    title_prefix : str
        Префикс для заголовков
    figsize : tuple
        Размер фигуры
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Добавляем флаг категориальности если его нет
    if 'is_categorical' not in importance_df.columns:
        importance_df['is_categorical'] = importance_df['feature'].isin(categorical_features)
    
    # ===== График 1: ТОП-10 Самых Важных Признаков =====
    top_features = importance_df.head(10)
    colors = ['coral' if is_cat else 'steelblue' for is_cat in top_features['is_categorical']]
    
    # Если есть std в данных, добавляем error bars
    if 'importance_std' in importance_df.columns:
        axes[0].barh(top_features['feature'], top_features['importance'], 
                     color=colors, edgecolor='black', 
                     xerr=top_features['importance_std'], capsize=5, alpha=0.8)
        xlabel = 'Feature Importance (Mean ± Std)'
    else:
        axes[0].barh(top_features['feature'], top_features['importance'], 
                     color=colors, edgecolor='black')
        xlabel = 'Feature Importance'
        
        # Добавляем значения на график
        for i, (feature, importance) in enumerate(zip(top_features['feature'], 
                                                       top_features['importance'])):
            axes[0].text(importance, i, f'{importance:.4f}', 
                        va='center', ha='left', fontsize=9, fontweight='bold')
    
    axes[0].set_xlabel(xlabel, fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Features', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{title_prefix}ТОП-10 Самых Важных Признаков\nCatBoost модель', 
                      fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Легенда
    legend_elements = [
        Patch(facecolor='steelblue', label='Числовые'),
        Patch(facecolor='coral', label='Категориальные')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')
    
    # ===== График 2: Кумулятивный Вклад =====
    importance_col = 'importance_mean' if 'importance_mean' in importance_df.columns else 'importance'
    importance_sorted = importance_df[importance_col].values
    cumulative_importance = np.cumsum(importance_sorted) / importance_sorted.sum() * 100
    
    axes[1].plot(range(len(cumulative_importance)), cumulative_importance, 
                 color='darkgreen', linewidth=3, label='Кумулятивный вклад (%)')
    axes[1].fill_between(range(len(cumulative_importance)), cumulative_importance, 
                          alpha=0.3, color='lightgreen')
    
    # Пороги 80% и 90%
    axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% порог')
    axes[1].axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% порог')
    
    n_features_80 = np.argmax(cumulative_importance >= 80) + 1
    n_features_90 = np.argmax(cumulative_importance >= 90) + 1
    
    axes[1].axvline(x=n_features_80, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].axvline(x=n_features_90, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    axes[1].text(n_features_80, 82, f'{n_features_80} признаков', 
                 fontsize=10, fontweight='bold', color='red', ha='left')
    axes[1].text(n_features_90, 92, f'{n_features_90} признаков', 
                 fontsize=10, fontweight='bold', color='orange', ha='left')
    
    axes[1].set_xlabel('Количество признаков', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Кумулятивный вклад (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Кумулятивный Вклад Признаков в Модель', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print(f"\n📊 АНАЛИЗ КУМУЛЯТИВНОГО ВКЛАДА ПРИЗНАКОВ:")
    print(f"  • Для достижения 80% важности требуется: {n_features_80} признаков "
          f"({n_features_80/len(X_columns)*100:.1f}%)")
    print(f"  • Для достижения 90% важности требуется: {n_features_90} признаков "
          f"({n_features_90/len(X_columns)*100:.1f}%)")
    print(f"  • Общее количество признаков: {len(X_columns)}")
    
    return {'n_features_80': n_features_80, 'n_features_90': n_features_90}


def optimize_threshold(y_true, y_pred_proba, cost_fp=10, cost_fn=100, 
                       thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    Оптимизация порога классификации по бизнес-метрикам
    
    Parameters:
    -----------
    y_true : array
        Истинные метки
    y_pred_proba : array
        Вероятности предсказаний
    cost_fp : float
        Стоимость False Positive (блокировка легальной транзакции)
    cost_fn : float
        Стоимость False Negative (пропущенный фрод)
    thresholds : array
        Массив порогов для проверки
        
    Returns:
    --------
    dict : Результаты оптимизации
    """
    print("\n" + "="*100)
    print("🎯 ОПТИМИЗАЦИЯ ПОРОГА КЛАССИФИКАЦИИ")
    print("="*100)
    
    threshold_metrics = {
        'threshold': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'business_cost': []
    }
    
    for threshold in thresholds:
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        cm_temp = confusion_matrix(y_true, y_pred_temp)
        
        if len(cm_temp.ravel()) == 4:
            tn, fp, fn, tp = cm_temp.ravel()
        else:
            continue
        
        business_cost = fp * cost_fp + fn * cost_fn
        
        threshold_metrics['threshold'].append(threshold)
        threshold_metrics['precision'].append(precision_score(y_true, y_pred_temp, zero_division=0))
        threshold_metrics['recall'].append(recall_score(y_true, y_pred_temp, zero_division=0))
        threshold_metrics['f1'].append(f1_score(y_true, y_pred_temp, zero_division=0))
        threshold_metrics['accuracy'].append(accuracy_score(y_true, y_pred_temp))
        threshold_metrics['business_cost'].append(business_cost)
    
    # Оптимальные пороги
    optimal_f1_idx = np.argmax(threshold_metrics['f1'])
    optimal_business_idx = np.argmin(threshold_metrics['business_cost'])
    
    optimal_threshold_f1 = thresholds[optimal_f1_idx]
    optimal_threshold_business = thresholds[optimal_business_idx]
    
    print(f"\n📊 ОПТИМАЛЬНЫЕ ПОРОГИ:")
    print(f"   По F1-Score:        {optimal_threshold_f1:.2f} "
          f"(F1={threshold_metrics['f1'][optimal_f1_idx]:.4f})")
    print(f"   По бизнес-стоимости: {optimal_threshold_business:.2f} "
          f"(Cost=${threshold_metrics['business_cost'][optimal_business_idx]:,.0f})")
    print(f"   Дефолтный порог:     0.50")
    
    return {
        'threshold_metrics': threshold_metrics,
        'optimal_f1': optimal_threshold_f1,
        'optimal_business': optimal_threshold_business,
        'cost_fp': cost_fp,
        'cost_fn': cost_fn
    }


def plot_threshold_analysis(threshold_results, figsize=(16, 6)):
    """
    Визуализация анализа порогов
    
    Parameters:
    -----------
    threshold_results : dict
        Результаты от optimize_threshold()
    figsize : tuple
        Размер фигуры
    """
    tm = threshold_results['threshold_metrics']
    thresholds = np.array(tm['threshold'])
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ===== График 1: Метрики vs Порог =====
    axes[0].plot(thresholds, tm['precision'], label='Precision', linewidth=2, marker='o', markersize=4)
    axes[0].plot(thresholds, tm['recall'], label='Recall', linewidth=2, marker='s', markersize=4)
    axes[0].plot(thresholds, tm['f1'], label='F1-Score', linewidth=2, marker='^', markersize=4)
    axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Default (0.5)')
    axes[0].axvline(x=threshold_results['optimal_f1'], color='green', linestyle='--', linewidth=2, 
                    label=f"Optimal F1 ({threshold_results['optimal_f1']:.2f})")
    axes[0].axvline(x=threshold_results['optimal_business'], color='purple', linestyle='--', linewidth=2, 
                    label=f"Optimal Cost ({threshold_results['optimal_business']:.2f})")
    axes[0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Метрики в зависимости от порога', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # ===== График 2: Бизнес-стоимость vs Порог =====
    optimal_cost = min(tm['business_cost'])
    axes[1].plot(thresholds, tm['business_cost'], color='red', linewidth=3, marker='o', markersize=6)
    axes[1].axvline(x=threshold_results['optimal_business'], color='green', linestyle='--', linewidth=2, 
                    label=f'Минимум (${optimal_cost:,.0f})')
    axes[1].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Business Cost ($)', fontsize=12, fontweight='bold')
    axes[1].set_title(f"Бизнес-стоимость ошибок vs Порог\n"
                      f"(FP=${threshold_results['cost_fp']}, FN=${threshold_results['cost_fn']})", 
                      fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(alpha=0.3)
    axes[1].ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.show()






# def plot_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
#     """
#     Визуализирует важность признаков из модели (для понимания, какие фичи работают).
    
#     Parameters:
#     -----------
#     model : sklearn model or xgb/lgb model
#         Обученная модель
#     feature_names : list
#         Имена признаков
#     top_n : int
#         Сколько топ фич показывать
#     title : str
#         Заголовок графика
#     """
#     # Получаем важности в зависимости от типа модели
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#     elif hasattr(model, 'coef_'):
#         importances = np.abs(model.coef_[0])
#     else:
#         print("Модель не поддерживает feature importance")
#         return
    
#     # Создаём dataframe и сортируем
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': importances
#     }).sort_values('importance', ascending=False).head(top_n)
    
#     fig, ax = plt.subplots(figsize=(10, top_n/2))
#     ax.barh(range(len(importance_df)), importance_df['importance'].values)
#     ax.set_yticks(range(len(importance_df)))
#     ax.set_yticklabels(importance_df['feature'].values)
#     ax.set_xlabel('Importance')
#     ax.set_title(f'{title} (Top {top_n})')
#     ax.invert_yaxis()
    
#     for i, v in enumerate(importance_df['importance'].values):
#         ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
#     plt.tight_layout()
#     plt.show()


# def plot_multiple_roc_curves(models_dict, X_test, y_test):
#     """
#     Сравнивает несколько моделей на одном ROC графике (для сравнения ensemble).
    
#     Parameters:
#     -----------
#     models_dict : dict
#         {model_name: model}
#     X_test : pd.DataFrame
#         Тестовые признаки
#     y_test : pd.Series
#         Тестовый таргет
#     """
#     plt.figure(figsize=(10, 8))
    
#     for model_name, model in models_dict.items():
#         y_pred_proba = model.predict_proba(X_test)[:, 1]
#         fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#         roc_auc = auc(fpr, tpr)
        
#         plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC-AUC Сравнение моделей')
#     plt.legend(loc="lower right")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
#     """
#     Матрица ошибок (для анализа TP, FP, TN, FN).
    
#     Parameters:
#     -----------
#     y_true : array-like
#         Истинные метки
#     y_pred : array-like
#         Предсказанные метки
#     model_name : str
#         Имя модели
#     """
#     cm = confusion_matrix(y_true, y_pred)
    
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
#                 xticklabels=['Non-Fraud', 'Fraud'],
#                 yticklabels=['Non-Fraud', 'Fraud'])
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(f'Confusion Matrix - {model_name}')
#     plt.tight_layout()
#     plt.show()


# def plot_cross_val_results(cv_results, metrics=['AUC', 'Precision', 'Recall', 'F1']):
#     """
#     Визуализирует результаты кросс-валидации (для понимания стабильности модели).
    
#     Parameters:
#     -----------
#     cv_results : dict
#         Результаты из cross_val_evaluate
#     metrics : list
#         Какие метрики показывать
#     """
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     axes = axes.flatten()
    
#     for i, metric in enumerate(metrics):
#         if metric in cv_results:
#             scores = cv_results[metric]['scores']
#             mean = cv_results[metric]['mean']
#             std = cv_results[metric]['std']
            
#             axes[i].bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
#             axes[i].axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
#             axes[i].fill_between(range(len(scores)), mean-std, mean+std, alpha=0.2, color='red')
#             axes[i].set_xlabel('Fold')
#             axes[i].set_ylabel(metric)
#             axes[i].set_title(f'{metric} по фолдам')
#             axes[i].legend()
#             axes[i].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()


# def plot_learning_curve(train_scores, val_scores, title="Learning Curve"):
#     """
#     Кривая обучения (для диагностики overfitting).
    
#     Parameters:
#     -----------
#     train_scores : list
#         Скоры на train на разных итерациях
#     val_scores : list
#         Скоры на validation на разных итерациях
#     title : str
#         Заголовок
#     """
#     iterations = range(1, len(train_scores) + 1)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, train_scores, 'b-', label='Train Score', linewidth=2)
#     plt.plot(iterations, val_scores, 'r-', label='Validation Score', linewidth=2)
#     plt.xlabel('Iteration')
#     plt.ylabel('AUC')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
    
def plot_dataset_overview(data, figsize=(14, 10)):
    """
    Визуализация обзора датасета: пропуски, уникальные значения, типы данных.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Датасет для визуализации
    figsize : tuple, default=(14, 10)
        Размер фигуры
    
    Returns:
    --------
    matplotlib.figure.Figure : Объект фигуры с графиками
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ========================================================================
    # График 1: Топ-20 признаков с пропусками
    # ========================================================================
    
    missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False).head(20)
    
    if len(missing_pct) > 0 and missing_pct.max() > 0:
        colors = ['red' if x > 80 else 'orange' if x > 50 else 'yellow' 
                  for x in missing_pct.values]
        
        axes[0, 0].barh(range(len(missing_pct)), missing_pct.values, 
                        color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_yticks(range(len(missing_pct)))
        axes[0, 0].set_yticklabels(missing_pct.index, fontsize=8)
        axes[0, 0].set_xlabel('% пропусков', fontsize=10, fontweight='bold')
        axes[0, 0].set_title('🔴 Топ-20: Признаки с пропусками', 
                             fontsize=11, fontweight='bold')
        axes[0, 0].axvline(x=80, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[0, 0].axvline(x=50, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x', alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Нет пропусков', ha='center', va='center',
                        fontsize=12, transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('🔴 Топ-20: Признаки с пропусками', 
                             fontsize=11, fontweight='bold')
    
    # ========================================================================
    # График 2: Распределение пропусков
    # ========================================================================
    
    missing_bins = pd.cut(data.isnull().sum(), 
                          bins=[-1, 0, 1000, 10000, 100000, float('inf')],
                          labels=['0', '1-1K', '1K-10K', '10K-100K', '>100K'])
    missing_dist = missing_bins.value_counts().sort_index()
    
    axes[0, 1].bar(range(len(missing_dist)), missing_dist.values, 
                   color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(range(len(missing_dist)))
    axes[0, 1].set_xticklabels(missing_dist.index, fontsize=9)
    axes[0, 1].set_ylabel('Количество признаков', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('📊 Распределение пропусков', fontsize=11, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(missing_dist.values):
        axes[0, 1].text(i, v, str(v), ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    
    # ========================================================================
    # График 3: Распределение уникальных значений
    # ========================================================================
    
    unique_bins = pd.cut(data.nunique(), 
                         bins=[-1, 2, 10, 100, 1000, float('inf')],
                         labels=['≤2', '3-10', '11-100', '101-1K', '>1K'])
    unique_dist = unique_bins.value_counts().sort_index()
    
    colors_unique = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    axes[1, 0].bar(range(len(unique_dist)), unique_dist.values, 
                   color=colors_unique[:len(unique_dist)], alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(len(unique_dist)))
    axes[1, 0].set_xticklabels(unique_dist.index, fontsize=9)
    axes[1, 0].set_ylabel('Количество признаков', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('🟢 Распределение уникальных', fontsize=11, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(unique_dist.values):
        axes[1, 0].text(i, v, str(v), ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    
    # ========================================================================
    # График 4: Типы данных
    # ========================================================================
    
    type_counts = data.dtypes.value_counts()
    axes[1, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                   colors=plt.cm.Set3(range(len(type_counts))), startangle=90)
    axes[1, 1].set_title('📈 Типы данных', fontsize=11, fontweight='bold')
    
    plt.suptitle('📊 КРАТКИЙ ОБЗОР ДАТАСЕТА', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# def analyze_feature_importance(pipe, X_train, save_csv=True, figsize=(16, 12)):
#     """
#     Анализ важности признаков модели из Pipeline.
    
#     Parameters:
#     -----------
#     pipe : sklearn.pipeline.Pipeline или imblearn.pipeline.Pipeline
#         Обученный Pipeline с моделью
#     X_train : pd.DataFrame
#         Датасет с признаками (для получения названий)
#     save_csv : bool, default=True
#         Сохранять ли результаты в CSV
#     figsize : tuple, default=(16, 12)
#         Размер фигуры с графиками
    
#     Returns:
#     --------
#     pd.DataFrame : DataFrame с feature importance
#     """
    
#     print("\n" + "="*100)
#     print("АНАЛИЗ FEATURE IMPORTANCE")
#     print("="*100)
    
#     # ========================================================================
#     # 1. Получение модели и важности
#     # ========================================================================
    
#     model = pipe.named_steps['model']
#     feature_importance = model.feature_importances_
    
#     # ========================================================================
#     # 2. Извлечение названий признаков
#     # ========================================================================
    
#     print("\n🔍 Извлечение названий признаков...")
    
#     # Пробуем получить названия из X_train
#     try:
#         feature_names = X_train.columns.tolist()
#         print(f"✓ Названия получены из X_train: {len(feature_names)} признаков")
#     except:
#         feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
#         print(f"⚠️ Используем индексы: {len(feature_names)} признаков")
    
#     # Проверка соответствия
#     if len(feature_names) != len(feature_importance):
#         print(f"⚠️ ВНИМАНИЕ: Несоответствие количества признаков!")
#         print(f"   feature_names: {len(feature_names)}")
#         print(f"   feature_importance: {len(feature_importance)}")
        
#         if len(feature_names) > len(feature_importance):
#             feature_names = feature_names[:len(feature_importance)]
#         else:
#             for i in range(len(feature_names), len(feature_importance)):
#                 feature_names.append(f'feature_{i}')
    
#     print(f"✓ Итого признаков для анализа: {len(feature_names)}")
    
#     # Создаём DataFrame
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': feature_importance
#     }).sort_values('importance', ascending=False)
    
#     # ========================================================================
#     # 3. ТОП-30 ВАЖНЫХ ПРИЗНАКОВ
#     # ========================================================================
    
#     print(f"\n📊 ТОП-30 САМЫХ ВАЖНЫХ ПРИЗНАКОВ:")
#     print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':<15} {'Cumulative %'}")
#     print(f"{'='*90}")
    
#     cumulative_importance = 0
#     total_importance = importance_df['importance'].sum()
    
#     for idx, (rank, row) in enumerate(importance_df.head(30).iterrows(), 1):
#         cumulative_importance += row['importance']
#         cumulative_pct = cumulative_importance / total_importance * 100
        
#         feature_display = row['feature'][:48] + '..' if len(row['feature']) > 50 else row['feature']
#         print(f"{idx:<6} {feature_display:<50} {row['importance']:<15.6f} {cumulative_pct:>6.2f}%")
    
#     print(f"\n💡 Топ-30 признаков объясняют {cumulative_pct:.2f}% важности модели")
    
#     # ========================================================================
#     # 4. СТАТИСТИКА
#     # ========================================================================
    
#     print(f"\n📈 СТАТИСТИКА:")
#     print(f"  Всего признаков:       {len(importance_df)}")
#     print(f"  Признаков с imp > 0:   {(importance_df['importance'] > 0).sum()}")
#     print(f"  Признаков с imp = 0:   {(importance_df['importance'] == 0).sum()}")
#     print(f"  Max importance:        {importance_df['importance'].max():.6f} ({importance_df.iloc[0]['feature']})")
#     print(f"  Mean importance:       {importance_df['importance'].mean():.6f}")
#     print(f"  Median importance:     {importance_df['importance'].median():.6f}")
    
#     # ========================================================================
#     # 5. ГРАФИКИ
#     # ========================================================================
    
#     fig, axes = plt.subplots(2, 2, figsize=figsize)
    
#     # График 1: ТОП-20
#     top_20 = importance_df.head(20).copy()
#     top_20['feature_short'] = top_20['feature'].apply(lambda x: x[:35] + '..' if len(x) > 37 else x)
    
#     axes[0, 0].barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.8)
#     axes[0, 0].set_yticks(range(len(top_20)))
#     axes[0, 0].set_yticklabels(top_20['feature_short'], fontsize=9)
#     axes[0, 0].invert_yaxis()
#     axes[0, 0].set_xlabel('Importance', fontsize=11, fontweight='bold')
#     axes[0, 0].set_title('Топ-20 признаков по важности', fontsize=13, fontweight='bold')
#     axes[0, 0].grid(axis='x', alpha=0.3)
    
#     # График 2: Cumulative Importance
#     importance_df_sorted = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
#     importance_df_sorted['cumulative'] = importance_df_sorted['importance'].cumsum() / importance_df_sorted['importance'].sum() * 100
    
#     axes[0, 1].plot(range(1, len(importance_df_sorted) + 1), importance_df_sorted['cumulative'], 
#                     color='coral', linewidth=2)
#     axes[0, 1].axhline(y=80, color='red', linestyle='--', linewidth=1, label='80%')
#     axes[0, 1].axhline(y=90, color='orange', linestyle='--', linewidth=1, label='90%')
#     axes[0, 1].axhline(y=95, color='green', linestyle='--', linewidth=1, label='95%')
#     axes[0, 1].set_xlabel('Количество признаков', fontsize=11, fontweight='bold')
#     axes[0, 1].set_ylabel('Cumulative Importance (%)', fontsize=11, fontweight='bold')
#     axes[0, 1].set_title('Накопительная важность', fontsize=13, fontweight='bold')
#     axes[0, 1].legend(loc='lower right')
#     axes[0, 1].grid(alpha=0.3)
    
#     n_80 = (importance_df_sorted['cumulative'] <= 80).sum() + 1
#     n_90 = (importance_df_sorted['cumulative'] <= 90).sum() + 1
#     n_95 = (importance_df_sorted['cumulative'] <= 95).sum() + 1
    
#     axes[0, 1].text(0.5, 0.3, f'{n_80} признаков → 80%\n{n_90} признаков → 90%\n{n_95} признаков → 95%',
#                     transform=axes[0, 1].transAxes, fontsize=11, 
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
#     # График 3: Распределение
#     axes[1, 0].hist(importance_df['importance'], bins=50, color='steelblue', 
#                     edgecolor='black', alpha=0.7)
#     axes[1, 0].axvline(x=importance_df['importance'].mean(), color='red', 
#                        linestyle='--', linewidth=2, label=f"Mean = {importance_df['importance'].mean():.6f}")
#     axes[1, 0].axvline(x=importance_df['importance'].median(), color='green', 
#                        linestyle='--', linewidth=2, label=f"Median = {importance_df['importance'].median():.6f}")
#     axes[1, 0].set_xlabel('Importance', fontsize=11, fontweight='bold')
#     axes[1, 0].set_ylabel('Количество признаков', fontsize=11, fontweight='bold')
#     axes[1, 0].set_title('Распределение Feature Importance', fontsize=13, fontweight='bold')
#     axes[1, 0].legend()
#     axes[1, 0].grid(alpha=0.3)
    
#     # График 4: Топ-15 с цветом
#     top_15 = importance_df.head(15).copy().sort_values('importance', ascending=True)
#     top_15['feature_short'] = top_15['feature'].apply(lambda x: x[:30] + '..' if len(x) > 32 else x)
#     colors = ['coral' if x > importance_df['importance'].median() else 'steelblue' for x in top_15['importance']]
    
#     axes[1, 1].barh(range(len(top_15)), top_15['importance'], color=colors, alpha=0.8)
#     axes[1, 1].set_yticks(range(len(top_15)))
#     axes[1, 1].set_yticklabels(top_15['feature_short'], fontsize=9)
#     axes[1, 1].set_xlabel('Importance', fontsize=11, fontweight='bold')
#     axes[1, 1].set_title('Топ-15 (цвет: выше/ниже медианы)', fontsize=13, fontweight='bold')
#     axes[1, 1].grid(axis='x', alpha=0.3)
    
#     legend_elements = [
#         Patch(facecolor='coral', alpha=0.8, label='Выше медианы'),
#         Patch(facecolor='steelblue', alpha=0.8, label='Ниже медианы')
#     ]
#     axes[1, 1].legend(handles=legend_elements, loc='lower right')
    
#     plt.tight_layout()
#     plt.show()
    
#     # ========================================================================
#     # 6. НИЗКОИНФОРМАТИВНЫЕ ПРИЗНАКИ
#     # ========================================================================
    
#     print(f"\n🗑️ НИЗКОИНФОРМАТИВНЫЕ ПРИЗНАКИ:")
    
#     zero_importance = importance_df[importance_df['importance'] == 0]
#     print(f"\nПризнаков с importance = 0: {len(zero_importance)}")
    
#     low_importance_threshold = total_importance * 0.001
#     low_importance = importance_df[importance_df['importance'] < low_importance_threshold]
#     print(f"Признаков с importance < 0.1%: {len(low_importance)}")
    
#     print(f"\n💡 РЕКОМЕНДАЦИИ:")
#     print(f"   - Можно удалить {len(zero_importance)} признаков с нулевой важностью")
#     print(f"   - Рассмотреть удаление {len(low_importance)} признаков с очень низкой важностью")
#     print(f"   - {n_80} признаков дают 80% важности → можно сократить модель")
    
#     # ========================================================================
#     # 7. СОХРАНЕНИЕ
#     # ========================================================================
    
#     if save_csv:
#         importance_df.to_csv('feature_importance.csv', index=False)
#         print(f"\n✅ Feature importance сохранён в 'feature_importance.csv'")
    
#     print(f"\n✅ АНАЛИЗ ЗАВЕРШЁН!")
    
#     return importance_df


