"""
helper.py - Вспомогательные функции для IEEE CIS Fraud Detection

Этот модуль содержит:
- Функции для загрузки и обработки данных
- Валидационные утилиты
- Функции для работы с временными рядами
- Метрики и оценка моделей
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import mutual_info_classif
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')
from typing import List


# ============================================================================
# ЗАГРУЗКА И БАЗОВАЯ ОБРАБОТКА ДАННЫХ
# ============================================================================

def load_data(transaction_path, identity_path):
    """
    Загружает транзакции и идентификационные данные, объединяет их.
    
    Parameters:
    -----------
    transaction_path : str
        Путь к CSV файлу с транзакциями
    identity_path : str
        Путь к CSV файлу с идентификационными данными
        
    Returns:
    --------
    pd.DataFrame
        Объединённый датасет
    """
    print("Загрузка данных...")
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    
    # Объединение по TransactionID
    data = transactions.merge(identity, on='TransactionID', how='left')
    print(f"Датасет загружен: {data.shape}")
    return data


import re

def get_feature_groups(data):
    """
    Разделяет признаки по типам для анализа и обработки.
    Работает с разделителями _ и -
    """
    def get_prefix(col):
        """Извлекает префикс столбца независимо от разделителя"""
        return re.match(r'^[a-zA-Z]+', col).group() if re.match(r'^[a-zA-Z]+', col) else ''
    
    v_features = [col for col in data.columns if get_prefix(col).upper() == 'V']
    c_features = [col for col in data.columns if get_prefix(col).upper() == 'C']
    d_features = [col for col in data.columns if get_prefix(col).upper() == 'D']
    m_features = [col for col in data.columns if get_prefix(col).upper() == 'M']
    card_features = [col for col in data.columns if get_prefix(col).lower() == 'card']
    addr_features = [col for col in data.columns if get_prefix(col).lower() == 'addr']
    id_features = [col for col in data.columns if get_prefix(col).lower() == 'id']
    
    return {
        'v_features': v_features,
        'c_features': c_features,
        'd_features': d_features,
        'm_features': m_features,
        'id_features': id_features,
        'card_features': card_features,
        'addr_features': addr_features
    }

def create_feature_groups(df):
    """Создает группы признаков по префиксам + base + target. Полное покрытие."""
    
    # Префиксы
    prefixes = ['id_', 'card', 'addr', 'C', 'D', 'M', 'V']
    
    # Группы по префиксам
    feature_groups = {}
    for prefix in prefixes:
        cols = sorted([col for col in df.columns if col.startswith(prefix)])
        if cols:
            feature_groups[f'{prefix}_features'] = cols
    
    # ✅ ИСКЛЮЧИТЬ DeviceInfo, DeviceType из D_features
    if 'D_features' in feature_groups:
        feature_groups['D_features'] = [col for col in feature_groups['D_features'] 
                                      if col not in ['DeviceInfo', 'DeviceType']]
    
    # Base + target
    base_cols = ['TransactionID', 'TransactionDT', 'TransactionAmt']
    target_col = 'isFraud'
    
    feature_groups['base_features'] = [col for col in base_cols if col in df.columns]
    feature_groups['target_features'] = [target_col] if target_col in df.columns else []
    
    # Остальное + DeviceInfo, DeviceType в 'other_features'
    grouped = set().union(*feature_groups.values())
    device_cols = [col for col in ['DeviceInfo', 'DeviceType'] if col in df.columns]
    other = sorted((set(df.columns) - grouped) | set(device_cols))
    feature_groups['other_features'] = other
    
    return feature_groups

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Общая сводка по пропускам в датасете"""
    mis = df.isna().sum()
    mis = mis[mis > 0].sort_values(ascending=False)
    res = pd.DataFrame({
        'Missing_Count': mis,
        'Missing_Percent': (mis / len(df) * 100).round(2)
    })
    return res

def group_missing_summary(df: pd.DataFrame, groups: dict[str, List[str]]) -> pd.DataFrame:
    """Сводка по пропускам по группам признаков"""
    rows = []
    n = len(df)
    
    for name, cols in sorted(groups.items()):
        cols_in_df = [c for c in cols if c in df.columns]
        if not cols_in_df:
            continue
            
        sub = df[cols_in_df]
        missing_count = sub.isna().sum().sum()
        total_cells = n * len(cols_in_df)
        affected_features = (sub.isna().sum() > 0).sum()
        
        rows.append({
            'Group': name,
            'Feature_Count': len(cols_in_df),
            'Missing_Count': int(missing_count),
            'Missing_Percent': round(missing_count / total_cells * 100, 2),
            'Affected_Features': affected_features,
        })
    
    return pd.DataFrame(rows).sort_values('Missing_Percent', ascending=False).reset_index(drop=True)



def split_feature_stats(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[features]

    # ===== ЧИСЛОВЫЕ =====
    num = sub.select_dtypes(include=['number'])
    if not num.empty:
        desc_num = num.describe().T
        missing_pct_num = num.isna().mean() * 100
        unique_num = num.nunique(dropna=True)

        num_stats = desc_num.copy()
        num_stats["Feature"] = num_stats.index
        num_stats["Type"] = num.dtypes
        num_stats["Missing_%"] = missing_pct_num.round(2)
        num_stats["Unique"] = unique_num

        num_stats = (
            num_stats[[
                "Feature", "Type", "Missing_%", "Unique",
                "mean", "std", "min", "25%", "50%", "75%", "max"
            ]]
            .rename(columns={
                "mean": "Mean",
                "std": "Std",
                "min": "Min",
                "max": "Max"
            })
            .reset_index(drop=True)
        )

        # округляем все числовые статистики до сотых
        num_cols_to_round = ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        num_stats[num_cols_to_round] = num_stats[num_cols_to_round].round(2)

    else:
        num_stats = pd.DataFrame(columns=[
            "Feature", "Type", "Missing_%", "Unique",
            "Mean", "Std", "Min", "25%", "50%", "75%", "Max"
        ])

    # ===== OBJECT / CATEGORY =====
    obj = sub.select_dtypes(include=['object', 'category'])
    if not obj.empty:
        desc_obj = obj.describe().T
        missing_pct_obj = obj.isna().mean() * 100

        obj_stats = desc_obj.copy()
        obj_stats["Feature"] = obj_stats.index
        obj_stats["Type"] = obj.dtypes
        obj_stats["Missing_%"] = missing_pct_obj.round(2)

        obj_stats = (
            obj_stats[[
                "Feature", "Type", "Missing_%", "count", "unique", "top", "freq"
            ]]
            .rename(columns={
                "count": "Count",
                "unique": "Unique",
                "top": "Top",
                "freq": "Freq"
            })
            .reset_index(drop=True)
        )
    else:
        obj_stats = pd.DataFrame(columns=[
            "Feature", "Type", "Missing_%", "Count", "Unique", "Top", "Freq"
        ])

    return num_stats, obj_stats



def run_fraud_feature_report(train: pd.DataFrame):
    """
    Запускает весь пайплайн из ноутбука:
    0) CONFIG (внутри функции)
    1) Stratified sample
    2) Corr (Pearson + Spearman)
    3) MI (top по corr + missing flags + median impute)
    4) D-features: mean/median diff + missing_rate
    5) Nonlinearity: fraud_rate range по квантильным бинам

    Возвращает словарь с основными результатами.
    """

    # -----------------------------
    # CONFIG
    # -----------------------------
    SEED = 42
    SAMPLE_SIZE = 100_000
    TOP_CORR_N = 200          # сколько взять по corr для MI
    MI_TOP_N = 30
    CORR_THR = 0.01
    N_BINS = 10               # для проверки нелинейности (бинирование)
    TARGET = "isFraud"
    ID_COL = "TransactionID"

    np.random.seed(SEED)

    # -----------------------------
    # HELPERS
    # -----------------------------
    def make_stratified_idx(y, n, seed=42):
        """Чтобы в sample гарантированно попали fraud'ы (иначе MI может плавать)."""
        n = min(n, len(y))
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
        idx, _ = next(sss.split(np.zeros(len(y)), y))
        return idx

    def pretty_head(df, n=20, title=None):
        if title:
            print("\n" + title)
            print("-" * len(title))
        if display is not None:
            display(df.head(n))
        else:
            print(df.head(n))

    def binned_fraud_rate(df, feat, y_col=TARGET, bins=10):
        s = df[[feat, y_col]].copy()
        s = s.dropna(subset=[feat])
        # если мало уникальных — qcut может упасть; обработаем
        try:
            s["bin"] = pd.qcut(s[feat], q=bins, duplicates="drop")
        except ValueError:
            return None
        g = s.groupby("bin")[y_col].agg(["count", "mean"]).rename(columns={"mean": "fraud_rate"})
        g["fraud_rate_pct"] = g["fraud_rate"] * 100
        return g.reset_index()

    # -----------------------------
    # PREP COLS
    # -----------------------------
    numeric_cols = (
        train.select_dtypes(include=[np.number])
             .columns
             .drop([ID_COL, TARGET], errors="ignore")
             .tolist()
    )

    y = train[TARGET].astype(int)
    idx = make_stratified_idx(y, SAMPLE_SIZE, seed=SEED)

    print(f"Rows: {len(train):,} | Fraud rate: {y.mean()*100:.2f}% | Sample: {len(idx):,}")

    # -----------------------------
    # 1) CORRELATIONS (Pearson + Spearman)
    # -----------------------------
    corr_pearson = train[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
    corr_spearman = train[numeric_cols].corrwith(y, method="spearman").abs().sort_values(ascending=False)

    corr_df = pd.DataFrame({
        "pearson_abs": corr_pearson,
        "spearman_abs": corr_spearman
    }).sort_values(["pearson_abs", "spearman_abs"], ascending=False)

    corr_filtered = corr_df[(corr_df["pearson_abs"] > CORR_THR) | (corr_df["spearman_abs"] > CORR_THR)]
    pretty_head(corr_filtered, 30, title=f"Top correlations (abs > {CORR_THR})")

    # -----------------------------
    # 2) MUTUAL INFORMATION (на top по corr + missing flags)
    # -----------------------------
    top_cols = corr_df.head(TOP_CORR_N).index.tolist()

    X = train[top_cols].iloc[idx].copy()
    y_s = y.iloc[idx].copy()

    # missing flags — почти всегда полезно на табличных fraud-датасетах
    X_miss = X.isna().astype(np.uint8).add_prefix("miss__")
    X = pd.concat([X, X_miss], axis=1)

    # простая стратегия: NaN -> median (или 0, но median обычно стабильнее)
    X = X.fillna(X.median(numeric_only=True))

    mi = mutual_info_classif(
        X, y_s,
        n_neighbors=5,
        random_state=SEED,
        n_jobs=-1
    )

    mi_df = (pd.DataFrame({"feature": X.columns, "mi": mi})
               .sort_values("mi", ascending=False)
               .reset_index(drop=True))

    pretty_head(mi_df, MI_TOP_N, title=f"Mutual Info top-{MI_TOP_N} (sample stratified, top-{TOP_CORR_N} corr + missing flags)")

    # -----------------------------
    # 3) D-features quick scan (mean/median + missing + effect size)
    # -----------------------------
    d_feats = [f"D{i}" for i in range(1, 16) if f"D{i}" in train.columns]

    if d_feats:
        tmp = train[[TARGET] + d_feats].copy()
        miss_rate = tmp[d_feats].isna().mean().rename("missing_rate")

        grp_mean = tmp.groupby(TARGET)[d_feats].mean().T.rename(columns={0: "mean_0", 1: "mean_1"})
        grp_med  = tmp.groupby(TARGET)[d_feats].median().T.rename(columns={0: "med_0", 1: "med_1"})

        d_summary = (grp_mean.join(grp_med)
                            .join(miss_rate)
                            .assign(
                                abs_mean_diff=lambda d: (d["mean_1"] - d["mean_0"]).abs(),
                                abs_med_diff =lambda d: (d["med_1"]  - d["med_0"]).abs()
                            )
                            .sort_values(["abs_med_diff", "abs_mean_diff"], ascending=False)
                            .reset_index()
                            .rename(columns={"index": "feature"}))

        pretty_head(d_summary, 15, title="D-features: mean/median diff + missing_rate")
    else:
        d_summary = None
        print("No D-features found")

    # -----------------------------
    # 4) НЕЛИНЕЙНОСТЬ: fraud_rate по бинам (quantiles)
    # -----------------------------
    # берём несколько лучших по MI (без miss__ фич)
    top_real_feats = [f for f in mi_df["feature"].head(10).tolist() if not f.startswith("miss__")]

    nonlinear_rows = []
    for f in top_real_feats:
        g = binned_fraud_rate(train[[f, TARGET]], f, TARGET, bins=N_BINS)
        if g is None:
            continue
        # простая метрика "нелинейности": разброс fraud rate по бинам
        fr_range = g["fraud_rate"].max() - g["fraud_rate"].min()
        nonlinear_rows.append((f, fr_range, g["count"].min(), g["count"].sum()))

    nonlinear_df = (pd.DataFrame(nonlinear_rows, columns=["feature", "fraud_rate_range", "min_bin_n", "total_n"])
                      .sort_values("fraud_rate_range", ascending=False))

    pretty_head(nonlinear_df, 10, title="Nonlinearity check: fraud_rate range across quantile bins (top MI feats)")

    return {
        "config": {
            "SEED": SEED, "SAMPLE_SIZE": SAMPLE_SIZE, "TOP_CORR_N": TOP_CORR_N, "MI_TOP_N": MI_TOP_N,
            "CORR_THR": CORR_THR, "N_BINS": N_BINS, "TARGET": TARGET, "ID_COL": ID_COL
        },
        "sample_idx": idx,
        "numeric_cols": numeric_cols,
        "corr_all": corr_df,
        "corr_filtered": corr_filtered,
        "mi": mi_df,
        "d_summary": d_summary,
        "nonlinear": nonlinear_df,
        "top_real_feats_for_nonlin": top_real_feats,
    }

def reduce_correlated_features(
    df: pd.DataFrame,
    features: list[str],
    corr_thresh: float = 0.75,
    group_by: str = "nan_count",   # "nan_count" (быстро) или "nan_pattern" (точнее)
    min_group_size: int = 2,
):
    
    """
    Умное удаление коррелирующих признаков с учётом паттерна пропусков.
    
    Алгоритм:
    1) Группирует признаки по структуре NaN (одинаковое число пропусков или идентичный паттерн)
    2) Внутри каждой группы находит компоненты связности по корреляции >corr_thresh
    3) В каждой компоненте оставляет признак с максимальным nunique (наиболее информативный)
    """
    
    # 1) Группируем по NaN-структуре
    groups = {}
    for col in features:
        s = df[col]
        if group_by == "nan_count":
            key = int(s.isna().sum())
        elif group_by == "nan_pattern":
            # хэш паттерна пропусков (чтобы не хранить огромные булевы вектора)
            key = int(pd.util.hash_pandas_object(s.isna(), index=False).sum())
        else:
            raise ValueError("group_by must be 'nan_count' or 'nan_pattern'")
        groups.setdefault(key, []).append(col)

    keep = []
    drop = []
    components_debug = []  # опционально: чтобы посмотреть, какие группы схлопнулись

    # 2) Внутри каждого блока режем по корреляции
    for _, cols in groups.items():
        if len(cols) < min_group_size:
            keep.extend(cols)
            continue

        corr = df[cols].corr().abs().fillna(0.0)
        cols_set = set(cols)
        visited = set()

        for c in cols:
            if c not in cols_set or c in visited:
                continue

            # BFS/DFS: находим компоненту связности по ребрам |corr|>thr
            stack = [c]
            comp = []
            visited.add(c)

            while stack:
                u = stack.pop()
                comp.append(u)

                neigh = corr.index[(corr.loc[u] > corr_thresh)].tolist()
                for v in neigh:
                    if v in cols_set and v not in visited:
                        visited.add(v)
                        stack.append(v)

            if len(comp) == 1:
                keep.append(comp[0])
                continue

            # 3) В компоненте оставляем фичу с максимальным nunique
            nunique = df[comp].nunique(dropna=True)
            best = nunique.idxmax()

            keep.append(best)
            to_drop = [x for x in comp if x != best]
            drop.extend(to_drop)

            components_debug.append({"component": comp, "kept": best, "dropped": to_drop})

    keep = sorted(set(keep), key=keep.index) if len(keep) else []
    drop = sorted(set(drop))
    return keep, drop, components_debug


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание временных признаков и их взаимодействий с суммой транзакции.
    
    Временные признаки критичны для fraud detection, так как:
    - Мошенники активнее ночью (меньше контроля, автоматизированные атаки)
    - Большие суммы ночью = подозрительный паттерн
    - День недели влияет на fraud rate (выходные vs будни)
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с признаками TransactionDT и TransactionAmt
    
    Возвращает:
    -----------
    df : pd.DataFrame
        Датафрейм с добавленными 12 временными признаками
    
    Создаваемые признаки:
    ---------------------
    Базовые временные (5):
    - hour_of_day : int (0-23) — час совершения транзакции
    - day_of_week : int (0-6) — день недели (0=понедельник, 6=воскресенье)
    - day_number : int — номер дня от начала периода
    - is_night : int (0/1) — флаг ночной транзакции (22:00-06:00)
    - is_early_morning : int (0/1) — флаг раннего утра (00:00-04:00)
    - is_weekend : int (0/1) — флаг выходного дня (сб-вс)
    - time_period : category — период дня (night/morning/afternoon/evening)
    
    Взаимодействия с TransactionAmt (5):
    - night_high_amount : int (0/1) — ночь × большая сумма (>$500)
    - TransactionAmt_log : float — log1p(TransactionAmt)
    - night_amount_log : float — ночь × log(сумма)
    - hour_amount_log : float — час × log(сумма)
    - suspicious_night_tx : int (0/1) — подозрительная ночная транзакция (>$200)
    - time_amt_category : category — период дня × категория суммы
    
    Пример использования:
    ---------------------
    train = create_temporal_features(train)
    test = create_temporal_features(test)
    
    print(f"Создано признаков: {train.shape[1]}")
    """
    
    # Проверка наличия необходимых столбцов
    if 'TransactionDT' not in df.columns or 'TransactionAmt' not in df.columns:
        print("⚠️ Пропущены TransactionDT или TransactionAmt — временные признаки не созданы")
        return df
    
    # Константы для преобразования времени
    SECONDS_PER_DAY = 86400   # 60 * 60 * 24
    SECONDS_PER_HOUR = 3600   # 60 * 60
    
    # =========================================================
    # БАЗОВЫЕ ВРЕМЕННЫЕ ПРИЗНАКИ
    # =========================================================
    
    # Час суток (0-23)
    # TransactionDT % 86400 = секунды внутри дня → делим на 3600 = часы
    df['hour_of_day'] = ((df['TransactionDT'] % SECONDS_PER_DAY) / SECONDS_PER_HOUR).astype(int)
    
    # День недели (0-6, где 0 = понедельник)
    # TransactionDT // 86400 = номер дня от начала → берём остаток от деления на 7
    df['day_of_week'] = (df['TransactionDT'] // SECONDS_PER_DAY) % 7
    
    # Абсолютный номер дня от начала периода
    # Полезно для выявления temporal drift (изменение fraud rate со временем)
    df['day_number'] = (df['TransactionDT'] // SECONDS_PER_DAY).astype(int)
    
    # =========================================================
    # ФЛАГИ ВРЕМЕНИ СУТОК
    # =========================================================
    
    # Ночь (22:00-06:00) — повышенный риск fraud
    # Мошенники активнее ночью: меньше контроля, автоматизированные атаки
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)
    
    # Раннее утро (00:00-04:00) — самый подозрительный период
    # Легитимные пользователи редко совершают транзакции в 2-4 часа ночи
    df['is_early_morning'] = (df['hour_of_day'] < 4).astype(int)
    
    # Выходные (суббота-воскресенье)
    # Fraud rate может отличаться в выходные vs будни
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Период дня (категориальный признак для деревьев)
    # night (00-06), morning (06-12), afternoon (12-18), evening (18-24)
    df['time_period'] = pd.cut(
        df['hour_of_day'],
        bins=[-1, 6, 12, 18, 24],  # границы периодов
        labels=['night', 'morning', 'afternoon', 'evening']
    ).astype('category')
    
    # =========================================================
    # ВЗАИМОДЕЙСТВИЯ С TransactionAmt
    # =========================================================
    
    # Ночь × большая сумма (>$500)
    # Паттерн: легитимные крупные покупки редко происходят ночью
    # Fraud часто использует украденные карты для больших сумм именно ночью
    df['night_high_amount'] = (
        df['is_night'] * (df['TransactionAmt'] > 500)
    ).astype(int)
    
    # Создаём log(TransactionAmt), если его ещё нет
    # log1p = log(1 + x) — избегаем log(0) и сжимаем выбросы
    if 'TransactionAmt_log' not in df.columns:
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    
    # Ночь × log(сумма)
    # Непрерывное взаимодействие: чем больше сумма ночью, тем выше значение
    df['night_amount_log'] = df['is_night'] * df['TransactionAmt_log']
    
    # Час × log(сумма)
    # Захватывает паттерн "в какое время какие суммы типичны"
    # Например: утром — маленькие (кофе), днём — средние (обеды), вечером — большие (рестораны)
    df['hour_amount_log'] = df['hour_of_day'] * df['TransactionAmt_log']
    
    # Подозрительная ночная транзакция (ночь + сумма >$200)
    # Более мягкий порог, чем night_high_amount
    # Ночью даже средние суммы подозрительны
    df['suspicious_night_tx'] = (
        (df['is_night'] == 1) & (df['TransactionAmt'] > 200)
    ).astype(int)
    
    # Комбинация период дня × категория суммы (категориальный признак)
    # Примеры: "night_vhigh", "morning_low", "afternoon_mid"
    # Для деревьев: позволяет учесть сложные нелинейные зависимости
    df['time_amt_category'] = (
        df['time_period'].astype(str) + '_' + 
        pd.cut(
            df['TransactionAmt'], 
            bins=[0, 50, 200, 1000, np.inf],  # границы категорий суммы
            labels=['low', 'mid', 'high', 'vhigh']
        ).astype(str)
    ).astype('category')
    
    # =========================================================
    # СТАТИСТИКА
    # =========================================================
    
    print("✅ Временные признаки + взаимодействия с TransactionAmt: 12 признаков")
    print(f"   📊 Базовые временные: 7 (hour, day_of_week, day_number, is_night, is_early_morning, is_weekend, time_period)")
    print(f"   📊 Взаимодействия: 5 (night_high_amount, night_amount_log, hour_amount_log, suspicious_night_tx, time_amt_category)")
    print(f"\n   🌙 Ночных транзакций: {df['is_night'].sum():,} ({df['is_night'].mean()*100:.1f}%)")
    print(f"   🌙 Ночных транзакций >$500: {df['night_high_amount'].sum():,} ({df['night_high_amount'].mean()*100:.2f}%)")
    print(f"   🌙 Подозрительных ночных: {df['suspicious_night_tx'].sum():,} ({df['suspicious_night_tx'].mean()*100:.2f}%)")
    print(f"   📅 Выходных транзакций: {df['is_weekend'].sum():,} ({df['is_weekend'].mean()*100:.1f}%)")
    
    return df


def create_transaction_amount_features(df: pd.DataFrame, quantiles_bins: list = None) -> tuple:
    """
    Создание производных признаков из TransactionAmt БЕЗ УТЕЧКИ данных.
    
    ВАЖНО: Для train вычисляем квантили, для test используем те же границы!
    Это предотвращает data leakage (утечку информации из test в train).
    
    Создаваемые признаки:
    1. TransactionAmt_log — log1p(сумма) для сжатия выбросов
    2. TransactionAmt_sqrt — sqrt(сумма) для сжатия выбросов (мягче log)
    3. TransactionAmt_bin — категории по фиксированным границам (micro/small/medium/high)
    4. TransactionAmt_decile — децили (0-9) для равномерного разбиения
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с признаком TransactionAmt
    quantiles_bins : list, optional
        Границы децилей, вычисленные на train (для применения к test)
        Если None, вычисляются автоматически (для train)
    
    Возвращает:
    -----------
    df : pd.DataFrame
        Датафрейм с добавленными 4 признаками
    decile_bins : list
        Границы децилей (для применения к test)
    
    Пример использования:
    ---------------------
    # Для train: вычисляем квантили
    train, train_decile_bins = create_transaction_amount_features(train)
    
    # Для test: используем те же границы
    test, _ = create_transaction_amount_features(test, quantiles_bins=train_decile_bins)
    
    ЗАЧЕМ НУЖНЫ ЭТИ ПРИЗНАКИ:
    -------------------------
    1. log/sqrt: TransactionAmt имеет выбросы ($0.25 до $31,937)
       - Логарифм сжимает большие суммы, делает распределение более нормальным
       - Полезно для линейных моделей (но для деревьев не обязательно)
    
    2. Бины (категории): модель может найти пороги
       - "micro" ($0-10): подарочные карты, мелкие покупки
       - "extreme" ($1000+): крупные покупки (требуют особого внимания)
    
    3. Децили: равномерное разбиение на 10 групп
       - Каждая группа содержит ~10% транзакций
       - Модель находит fraud rate для каждой группы
    """
    
    # Проверка наличия TransactionAmt
    if 'TransactionAmt' not in df.columns:
        print("⚠️ Колонка TransactionAmt не найдена")
        return df, None
    
    # =========================================================
    # 1. ЛОГАРИФМ (log1p для избежания log(0))
    # =========================================================
    # log1p(x) = log(1 + x) — безопасная версия log, работает с нулями
    # Сжимает большие суммы: $10,000 → 9.21, $1,000 → 6.91, $100 → 4.62
    # Полезно для линейных моделей, для деревьев не критично
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    
    # =========================================================
    # 2. КВАДРАТНЫЙ КОРЕНЬ
    # =========================================================
    # sqrt(x) — мягче сжимает выбросы, чем log
    # Пример: $10,000 → 100, $1,000 → 31.62, $100 → 10
    # Иногда работает лучше log для транзакционных данных
    df['TransactionAmt_sqrt'] = np.sqrt(df['TransactionAmt'])
    
    # =========================================================
    # 3. КАТЕГОРИИ ПО ФИКСИРОВАННЫМ ГРАНИЦАМ (bins)
    # =========================================================
    # Разбиваем суммы на осмысленные категории по бизнес-логике:
    # - micro ($0-10): мелкие покупки (кофе, снеки, подарочные карты)
    # - small ($10-50): обычные покупки (фастфуд, такси)
    # - medium ($50-100): средние покупки (одежда, книги)
    # - medium_high ($100-200): крупные покупки (электроника начального уровня)
    # - high ($200-500): дорогие покупки (смартфоны, планшеты)
    # - very_high ($500-1000): очень дорогие покупки (ноутбуки, ювелирка)
    # - extreme ($1000+): экстремально дорогие (мебель, бытовая техника)
    #
    # ВАЖНО: границы фиксированные (не зависят от данных) → нет утечки
    df['TransactionAmt_bin'] = pd.cut(
        df['TransactionAmt'], 
        bins=[0, 10, 50, 100, 200, 500, 1000, 10000],
        labels=['micro', 'small', 'medium', 'medium_high', 'high', 'very_high', 'extreme']
    ).astype('category')
    
    # =========================================================
    # 4. ДЕЦИЛИ (10 равных групп по размеру)
    # =========================================================
    # Разбиваем TransactionAmt на 10 групп, каждая содержит ~10% транзакций
    # Это позволяет модели найти fraud rate для каждой децили
    #
    # КРИТИЧЕСКИ ВАЖНО: для train вычисляем квантили, для test используем те же!
    # Иначе будет data leakage (утечка информации из test в train)
    
    if quantiles_bins is None:
        # Для TRAIN: вычисляем квантили (границы децилей)
        quantiles = df['TransactionAmt'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).tolist()
        decile_bins = [df['TransactionAmt'].min() - 0.01] + quantiles + [df['TransactionAmt'].max() + 1]
    else:
        # Для TEST: используем границы, вычисленные на train
        decile_bins = quantiles_bins
    
    # Применяем разбиение на децили
    # labels=False → возвращает номера групп (0, 1, 2, ..., 9)
    # duplicates='drop' → если границы совпадают (редко), удаляем дубликаты
    # include_lowest=True → включаем минимальное значение в первую группу
    df['TransactionAmt_decile'] = pd.cut(
        df['TransactionAmt'], 
        bins=decile_bins, 
        labels=False,
        duplicates='drop',
        include_lowest=True
    )
    
    # =========================================================
    # СТАТИСТИКА
    # =========================================================
    print("✅ TransactionAmt: добавлено 4 производных признака (БЕЗ УТЕЧКИ)")
    print(f"   📊 TransactionAmt_log: min={df['TransactionAmt_log'].min():.2f}, max={df['TransactionAmt_log'].max():.2f}")
    print(f"   📊 TransactionAmt_sqrt: min={df['TransactionAmt_sqrt'].min():.2f}, max={df['TransactionAmt_sqrt'].max():.2f}")
    print(f"   📊 TransactionAmt_bin: {df['TransactionAmt_bin'].nunique()} категорий")
    print(f"      Распределение: {df['TransactionAmt_bin'].value_counts().to_dict()}")
    print(f"   📊 TransactionAmt_decile: 10 групп (0-9)")
    
    # Показываем первые 5 границ децилей для понимания
    if quantiles_bins is None:
        decile_labels = [f'${x:.2f}' for x in decile_bins[:5]]
        print(f"      Децильные границы (первые 5): {decile_labels}... (всего {len(decile_bins)} границ)")
    else:
        print(f"      Использованы границы из train (всего {len(decile_bins)} границ)")
    
    print(f"\n🎯 Итого признаков в датасете: {df.shape[1]}")
    
    return df, decile_bins



def get_categorical_features(df):
    """
    Возвращает список категориальных признаков для CatBoost
    на основе официальной документации IEEE-CIS Fraud Detection
    """
    # Официальные категориальные признаки IEEE-CIS
    official_categorical = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo',
        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
        'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
        'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35',
        'id_36', 'id_37', 'id_38'
    ]
    
    categorical_features = [col for col in official_categorical if col in df.columns]
    
    # Добавляем колонки с dtype='category'
    for col in df.columns:
        if df[col].dtype.name == 'category' and col not in categorical_features:
            categorical_features.append(col)
            print(f"   ⚠️  Найдена дополнительная category колонка: {col}")
    
    # Добавляем колонки с dtype='object'
    for col in df.columns:
        if df[col].dtype == 'object' and col not in categorical_features:
            categorical_features.append(col)
            print(f"   ⚠️  Найдена object (string) колонка: {col}")
    
    return categorical_features


def prepare_categorical_features(X, categorical_features):
    """
    Преобразует категориальные признаки в строки и обрабатывает NaN
    
    Parameters:
    -----------
    X : DataFrame
        Датафрейм с признаками
    categorical_features : list
        Список категориальных признаков
        
    Returns:
    --------
    X_processed : DataFrame
        Обработанный датафрейм
    categorical_features : list
        Обновленный список категориальных признаков
    """
    print(f"\n🔧 Преобразование категориальных и проблемных признаков...")
    
    X_processed = X.copy()
    
    # Преобразуем категориальные признаки
    for col in categorical_features:
        if col in X_processed.columns:
            # Если колонка category dtype, преобразуем в str
            if X_processed[col].dtype.name == 'category':
                X_processed[col] = X_processed[col].astype(str)
            
            # Заменяем NaN на 'missing' и приводим к str
            X_processed[col] = X_processed[col].fillna('missing').astype(str)
    
    # Дополнительная проверка: ищем проблемные колонки со смешанными типами
    print(f"\n🔍 Проверка смешанных типов данных...")
    for col in X_processed.columns:
        if col not in categorical_features:
            if X_processed[col].dtype == 'object':
                print(f"   ⚠️  ВНИМАНИЕ! Колонка '{col}' имеет dtype='object', но не в категориальных")
                print(f"      Примеры значений: {X_processed[col].dropna().head(3).tolist()}")
                
                # Пробуем преобразовать в float
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                    print(f"      ✅ Успешно преобразовано в числовой тип")
                except:
                    # Если не получается - добавляем в категориальные
                    categorical_features.append(col)
                    X_processed[col] = X_processed[col].fillna('missing').astype(str)
                    print(f"      ⚠️  Не удалось преобразовать в числовой - добавлено в категориальные")
    
    print(f"\n✅ Преобразовано {len(categorical_features)} категориальных признаков")
    
    return X_processed, categorical_features



# ============================================================================
# ВАЛИДАЦИОННЫЕ СТРАТЕГИИ
# ============================================================================

def get_time_series_split(data, n_splits=5):
    """
    TimeSeriesSplit для соблюдения временного порядка.
    Важно: более ранние данные в train, поздние в test.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Датасет (должен быть отсортирован по времени)
    n_splits : int
        Количество фолдов
        
    Returns:
    --------
    list of tuples
        [(train_idx, test_idx), ...]
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tss.split(data))
    print(f"TimeSeriesSplit: {n_splits} фолдов")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Фолд {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    return splits



# ============================================================================
# ОЦЕНКА МОДЕЛЕЙ
# ============================================================================

def evaluate_model(y_true, y_pred_proba, y_pred=None, threshold=0.5):
    """
    Вычисляет метрики классификации.
    
    Parameters:
    -----------
    y_true : array-like
        Истинные метки
    y_pred_proba : array-like
        Вероятности класса 1
    y_pred : array-like, optional
        Предсказанные метки (вычисляются если не даны)
    threshold : float
        Порог для бинаризации вероятностей
        
    Returns:
    --------
    dict
        Словарь с метриками
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    
    return metrics


def cross_val_evaluate(model, X, y, cv, stratified=False):
    """
    Кросс-валидация с вычислением метрик.
    
    Parameters:
    -----------
    model : sklearn model
        Модель для оценки
    X : pd.DataFrame
        Признаки
    y : pd.Series
        Таргет
    cv : int or cross-validator
        Стратегия кросс-валидации
    stratified : bool
        Использовать StratifiedKFold?
        
    Returns:
    --------
    dict
        Средние метрики по фолдам
    """
    scoring = {
        'auc': 'roc_auc',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    result = {}
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        result[metric] = {
            'mean': test_scores.mean(),
            'std': test_scores.std(),
            'scores': test_scores
        }
    
    return result


# ============================================================================
# РАБОТА С ВРЕМЕНЕМ
# ============================================================================

def extract_time_features(data, datetime_col='TransactionDT'):
    """
    Извлекает временные признаки из Timedelta колонки.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Датасет
    datetime_col : str
        Имя колонки с временем
        
    Returns:
    --------
    pd.DataFrame
        Датасет с новыми временными признаками
    """
    data_copy = data.copy()
    
    # TransactionDT в секундах, конвертируем в часы и дни
    if datetime_col in data_copy.columns:
        data_copy['hour'] = (data_copy[datetime_col] // 3600) % 24
        data_copy['day_of_week'] = (data_copy[datetime_col] // 86400) % 7
        data_copy['day_of_month'] = (data_copy[datetime_col] // 86400) % 30
        
    return data_copy


def get_sorted_by_time(data, time_col='TransactionDT'):
    """
    Сортирует данные по времени (для TimeSeriesSplit).
    
    Parameters:
    -----------
    data : pd.DataFrame
    time_col : str
        Колонка со временем
        
    Returns:
    --------
    pd.DataFrame
        Отсортированный датасет
    """
    return data.sort_values(by=time_col).reset_index(drop=True)


# ============================================================================
# АНАЛИЗ ДИСБАЛАНСА
# ============================================================================

def analyze_class_balance(y, name=""):
    """
    Анализирует дисбаланс классов.
    
    Parameters:
    -----------
    y : pd.Series or array-like
        Таргет
    name : str
        Название для вывода
    """
    counts = pd.Series(y).value_counts()
    proportions = pd.Series(y).value_counts(normalize=True) * 100
    
    print(f"\n{name} Дисбаланс классов:")
    for cls in sorted(counts.index):
        print(f"  Класс {cls}: {counts[cls]} ({proportions[cls]:.2f}%)")


# ============================================================================
# УТИЛИТЫ ДЛЯ FEATURE ENGINEERING
# ============================================================================

def create_uid(data, cols=['card1', 'addr1', 'D1']):
    """
    Создаёт Unique Identifier для агрегационного кодирования.
    Это главная "магия" выигрывающего решения!
    
    Parameters:
    -----------
    data : pd.DataFrame
    cols : list
        Колонки для создания UID
        
    Returns:
    --------
    pd.Series
        UID для каждой строки
    """
    data_copy = data.copy()
    for col in cols:
        if col not in data_copy.columns:
            print(f"Внимание: колонка {col} не найдена")
    
    # Конкатенируем с разделителем
    uid = data_copy[cols[0]].astype(str)
    for col in cols[1:]:
        uid = uid + '_' + data_copy[col].astype(str)
    
    return uid


def frequency_encode(data, col):
    """
    Фреквентное кодирование категориальной переменной.
    Считает, как часто встречается каждое значение.
    
    Parameters:
    -----------
    data : pd.DataFrame
    col : str
        Колонка для кодирования
        
    Returns:
    --------
    pd.Series
        Фреквентные коды
    """
    freq_map = data[col].value_counts().to_dict()
    return data[col].map(freq_map).fillna(0).astype(int)


def handle_missing_values(data, d_features, fill_value=-1):
    """
    Обработка пропущенных значений в D-признаках.
    D признаки - это timedeltas, пропуск значит первая транзакция.
    
    Parameters:
    -----------
    data : pd.DataFrame
    d_features : list
        Список D признаков
    fill_value : int
        Значение для заполнения
        
    Returns:
    --------
    pd.DataFrame
    """
    data_copy = data.copy()
    for col in d_features:
        if col in data_copy.columns:
            data_copy[col].fillna(fill_value, inplace=True)
    
    return data_copy


def analyze_object_columns(data):
    """Анализ всех столбцов типа object"""
    object_cols = data.select_dtypes(include='object').columns
    
    print(f"Всего столбцов object: {len(object_cols)}\n")
    
    for col in object_cols:
        print(f"{'='*60}")
        print(f"Столбец: {col}")
        print(f"{'='*60}")
        print(f"Тип данных: {data[col].dtype}")
        print(f"Всего значений: {len(data[col])}")
        print(f"Уникальных значений: {data[col].nunique()}")
        print(f"Пропусков: {data[col].isnull().sum()} ({data[col].isnull().sum() / len(data) * 100:.2f}%)")
        print(f"Самые частые значения:")
        print(data[col].value_counts().head(5))
        print()
        
def analyze_numeric_columns(data):
    """Анализ всех числовых столбцов (int и float)"""
    numeric_cols = data.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns
    
    print(f"Всего числовых столбцов: {len(numeric_cols)}\n")
    
    for col in numeric_cols:
        print(f"{'='*60}")
        print(f"Столбец: {col} ({data[col].dtype})")
        print(f"{'='*60}")
        print(f"Всего значений: {len(data[col])}")
        print(f"Уникальных значений: {data[col].nunique()}")
        print(f"Пропусков: {data[col].isnull().sum()} ({data[col].isnull().sum() / len(data) * 100:.2f}%)")
        print(f"Min: {data[col].min():.4f}")
        print(f"Max: {data[col].max():.4f}")
        print(f"Mean: {data[col].mean():.4f}")
        print(f"Median: {data[col].median():.4f}")
        print(f"Std: {data[col].std():.4f}")
        print()


# ============================================================================
# PRINT УТИЛИТЫ
# ============================================================================

def print_section(title, char='='):
    """Печатает красивый заголовок раздела."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


def print_results(results_dict, metric_name="Метрика"):
    """Печатает результаты моделей в табличном формате."""
    print(f"\n{metric_name}:")
    for model_name, metrics in results_dict.items():
        print(f"\n  {model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"    {metric}: {value['mean']:.4f} (+/- {value['std']:.4f})")
            else:
                print(f"    {metric}: {value:.4f}")
                
def analyze_missing_by_groups(data):
    """Анализ пропусков по группам признаков"""
    feature_groups = get_feature_groups(data)
    
    results = []
    for group_name, features in feature_groups.items():
        if features:  # Если группа не пуста
            group_data = data[features]
            missing_count = group_data.isnull().sum().sum()
            missing_percent = (missing_count / (len(data) * len(features))) * 100
            
            results.append({
                'Group': group_name,
                'Feature_Count': len(features),
                'Missing_Count': missing_count,
                'Missing_Percent': round(missing_percent, 2),
                'Affected_Features': (group_data.isnull().sum() > 0).sum()
            })
    
    return pd.DataFrame(results).sort_values('Missing_Percent', ascending=False)



def get_dataset_stats(data):
    """
    Анализ датасета: статистика и выявление проблемных признаков.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Датасет для анализа
    
    Returns:
    --------
    dict : Словарь с результатами анализа
        - to_drop_high_missing: список признаков с >80% пропусков
        - to_drop_constants: список константных признаков
        - to_check: список признаков для проверки (50-80% пропусков)
        - binary_features: список бинарных признаков
        - categorical_features: список категориальных признаков
        - numeric_features: список числовых признаков
        - possible_ids: список возможных ID
        - stats: общая статистика
    """
    
    print("="*100)
    print("КРАТКАЯ СТАТИСТИКА ПО ПРИЗНАКАМ")
    print("="*100)
    
    # Разделение на типы
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📊 Всего признаков:   {data.shape[1]}")
    print(f"📊 Всего строк:       {data.shape[0]:,}")
    print(f"📈 Числовых:          {len(numeric_cols)}")
    print(f"📝 Категориальных:    {len(categorical_cols)}")
    
    # Ключевые метрики
    print("\n" + "="*100)
    print("📋 КЛЮЧЕВЫЕ МЕТРИКИ")
    print("="*100 + "\n")
    
    total_missing = data.isnull().sum().sum()
    total_cells = data.shape[0] * data.shape[1]
    
    print(f"{'Метрика':<40} {'Значение':<20}")
    print("-"*60)
    print(f"{'Признаков с пропусками':<40} {data.isnull().any().sum():<20}")
    print(f"{'Признаков без пропусков':<40} {data.shape[1] - data.isnull().any().sum():<20}")
    print(f"{'Общий % пропусков':<40} {total_missing / total_cells * 100:.2f}%")
    print(f"{'Признаков с <10 уникальных':<40} {sum(data.nunique() < 10):<20}")
    print(f"{'Признаков с >1000 уникальных':<40} {sum(data.nunique() > 1000):<20}")
    
    # Проблемные признаки
    print("\n" + "="*100)
    print("⚠️ ПРОБЛЕМНЫЕ ПРИЗНАКИ")
    print("="*100)
    
    # Пропуски > 80%
    high_missing = data.columns[data.isnull().sum() / len(data) > 0.8].tolist()
    print(f"\n🔴 Пропусков >80% ({len(high_missing)} признаков):")
    if high_missing:
        for col in high_missing[:10]:
            pct = data[col].isnull().sum() / len(data) * 100
            print(f"   {col:<35} {pct:>6.2f}%")
        if len(high_missing) > 10:
            print(f"   ... и ещё {len(high_missing) - 10} признаков")
    else:
        print("   ✅ Нет")
    
    # Пропуски 50-80%
    medium_missing = data.columns[(data.isnull().sum() / len(data) > 0.5) & 
                                   (data.isnull().sum() / len(data) <= 0.8)].tolist()
    print(f"\n🟠 Пропусков 50-80% ({len(medium_missing)} признаков):")
    if medium_missing:
        for col in medium_missing[:10]:
            pct = data[col].isnull().sum() / len(data) * 100
            print(f"   {col:<35} {pct:>6.2f}%")
        if len(medium_missing) > 10:
            print(f"   ... и ещё {len(medium_missing) - 10} признаков")
    else:
        print("   ✅ Нет")
    
    # Константы
    constants = data.columns[data.nunique() == 1].tolist()
    print(f"\n⚪ Константы (1 уникальное) ({len(constants)} признаков):")
    if constants:
        for col in constants:
            print(f"   {col}")
    else:
        print("   ✅ Нет")
    
    # Бинарные
    binary = data.columns[data.nunique() == 2].tolist()
    print(f"\n🟡 Бинарные (2 уникальных) ({len(binary)} признаков)")
    
    # Возможные ID
    possible_ids = []
    if len(categorical_cols) > 0:
        possible_ids = [col for col in categorical_cols if data[col].nunique() / len(data) > 0.9]
        print(f"\n🔵 Возможные ID (>90% уникальных) ({len(possible_ids)} признаков):")
        if possible_ids:
            for col in possible_ids:
                unique_pct = data[col].nunique() / len(data) * 100
                print(f"   {col:<35} {unique_pct:>6.2f}% уникальных")
        else:
            print("   ✅ Нет")
    
    # Выводы
    print("\n" + "="*100)
    print("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("="*100 + "\n")
    
    to_drop_80 = data.columns[data.isnull().sum() / len(data) > 0.8].tolist()
    to_drop_const = data.columns[data.nunique() == 1].tolist()
    
    print(f"❌ УДАЛИТЬ ({len(to_drop_80) + len(to_drop_const)} признаков):")
    print(f"   - Пропусков >80%: {len(to_drop_80)} признаков")
    print(f"   - Константы: {len(to_drop_const)} признаков")
    
    to_check = data.columns[(data.isnull().sum() / len(data) > 0.5) & 
                            (data.isnull().sum() / len(data) <= 0.8)].tolist()
    print(f"\n⚠️ ПРОВЕРИТЬ ВАЖНОСТЬ ({len(to_check)} признаков):")
    print(f"   - Пропусков 50-80%: {len(to_check)} признаков")
    
    print(f"\n✅ ОБРАБОТАТЬ:")
    print(f"   - Бинарные (2 уникальных): {len(binary)} признаков → Закодировать 0/1")
    print(f"   - Категориальные: {len(categorical_cols)} признаков → Label encoding или One-hot")
    print(f"   - Пропуски <50%: создать флаги _is_missing, заполнить медианой/модой")
    
    print("\n✅ АНАЛИЗ ЗАВЕРШЁН!")
    
    # Возврат результатов
    results = {
        'to_drop_high_missing': to_drop_80,
        'to_drop_constants': to_drop_const,
        'to_check': to_check,
        'binary_features': binary,
        'categorical_features': categorical_cols,
        'numeric_features': numeric_cols,
        'possible_ids': possible_ids,
        'stats': {
            'total_features': data.shape[1],
            'total_rows': data.shape[0],
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'features_with_missing': data.isnull().any().sum(),
            'total_missing_pct': total_missing / total_cells * 100,
            'binary_features': len(binary),
            'constant_features': len(constants)
        }
    }
    
    return results

def eda_cat_fraud_report(
    df,
    feats,
    target='isFraud',
    top_n=10,
    min_nobs=100,
    top_fraud_n=8,
    lift_thr=1.5,
    verbose=True
):
    """
    Печатает EDA-отчет по категориальным фичам (и любым дискретным/группируемым),
    считает fraud-rate по категориям и lift относительно baseline.
    
    Возвращает summary DataFrame с метриками по фичам.
    """
    baseline = df[target].mean() * 100

    if verbose:
        print("🔍 EDA КАТЕГОРИАЛЬНЫХ ФИЧ + DIST (Fraud Rate анализ)")
        print("=" * 80)

    rows = []

    for feat in feats:
        if feat not in df.columns:
            if verbose:
                print(f"❌ {feat} отсутствует")
            continue

        s = df[feat]
        vc = s.value_counts(dropna=False)

        # crosstab процентов
        ct = pd.crosstab(s, df[target], normalize='index') * 100  # проценты по строкам [web:101]
        ct['nobs'] = vc

        # оставляем только категории с достаточным nobs
        top_fraud = (
            ct[ct['nobs'] >= min_nobs]
            .sort_values(1, ascending=False)
            .head(top_fraud_n)
        )

        max_fraud = float(top_fraud[1].max()) if len(top_fraud) else np.nan
        lift = (max_fraud / baseline) if (baseline > 0 and not np.isnan(max_fraud)) else np.nan

        rec = "🔥 СИЛЬНАЯ фича" if (not np.isnan(lift) and lift > lift_thr) else "✅ Хорошая/слабая"

        miss_rate = s.isna().mean()
        nunique = s.nunique(dropna=True)

        rows.append({
            "feature": feat,
            "baseline_fraud_%": baseline,
            "max_fraud_%": max_fraud,
            "lift": lift,
            "missing_rate": miss_rate,
            "nunique": nunique,
            "recommendation": rec,
            "top_fraud_table": top_fraud.round(2)[[0, 1]] if len(top_fraud) else None,
            "top_counts": vc.head(top_n)  # можно потом смотреть
        })

        if verbose:
            print(f"\n📊 {feat.upper()}")
            print("-" * 50)
            print(f"Топ-{top_n}:\n{vc.head(top_n)}")

            if len(top_fraud):
                print(f"\nFraud % (топ, >={min_nobs} obs):\n{top_fraud.round(2)[[0,1]]}")
                print(f"📈 Рекомендация: {rec} (max {max_fraud:.1f}% vs baseline {baseline:.1f}%, lift x{lift:.1f})")
            else:
                print(f"\nFraud %: нет категорий с nobs >= {min_nobs}")
                print(f"📈 Рекомендация: {rec} (baseline {baseline:.1f}%)")

            print(f"Missing: {miss_rate:.1%} | Unique: {nunique}")

    summary = pd.DataFrame(rows).sort_values("lift", ascending=False, na_position="last")

    if verbose:
        print("\n🎯 ИТОГ: Target encode топ-3 по lift + is_missing_* бинарки")

    return summary

def plot_corr_and_pairs(
    df,
    features,
    title='Корреляции',
    threshold=0.9,
    max_pairs=10,
    figsize=(10, 8),
    plot=True,
    print_pairs=True
):
    corr = df[features].corr()

    # 1) Heatmap
    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr, cmap='coolwarm', center=0, square=True,
            cbar_kws={'label': 'Корреляция'}
        )
        plt.title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # 2) Пары |r| > threshold (верхний треугольник)
    cols = corr.columns
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pairs.append((cols[i], cols[j], float(r)))

    pairs_df = (pd.DataFrame(pairs, columns=['feat_1', 'feat_2', 'corr'])
                .sort_values('corr', key=lambda s: s.abs(), ascending=False))

    if print_pairs:
        print(f'\nВысоко коррелированные пары (|r| > {threshold}):')
        if len(pairs_df):
            for _, row in pairs_df.head(max_pairs).iterrows():
                print(f"{row['feat_1']} <-> {row['feat_2']}: {row['corr']:.4f}")
        else:
            print('Нет пар выше порога')

    # возвращаем только пары (не матрицу)
    return pairs_df