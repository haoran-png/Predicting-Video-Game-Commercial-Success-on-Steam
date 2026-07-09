import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint

def load_data(data_path):
    """
    Loads the CSV dataset and parses the 'Release date' column as datetime.
    """
    df = pd.read_csv(data_path)
    if 'Release date' in df.columns:
        df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
        # Drop rows with invalid/missing release dates to ensure clean temporal operations
        df = df.dropna(subset=['Release date'])
    return df

def vectorized_wilson_lower_bound(pos, neg, alpha=0.05):
    """
    Vectorized calculation of the Wilson confidence interval lower bound.
    Avoids slow pandas apply(axis=1) by processing columns directly as arrays.
    """
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    n = pos + neg
    
    # Use temporary arrays to prevent division-by-zero runtime warnings
    n_safe = np.where(n == 0, 1, n)
    pos_safe = np.where(n == 0, 0, pos)
    
    lower_bound, _ = proportion_confint(pos_safe, n_safe, alpha=alpha, method='wilson')
    
    # Assign baseline lower bound of 0.0 to games with zero reviews
    return np.where(n == 0, 0.0, lower_bound)

def calculate_success_score(df):
    """
    Calculates Review_Count, wilson_lb, and the target success_score.
    """
    df = df.copy()
    df['Review_Count'] = df['Positive'] + df['Negative']
    df['wilson_lb'] = vectorized_wilson_lower_bound(df['Positive'], df['Negative'])
    df['success_score'] = np.log1p(df['Review_Count']) * df['wilson_lb']
    return df

def filter_zero_reviews(df):
    """
    Drops games with zero reviews (Review_Count == 0), as they cannot have
    a meaningful target variable.
    """
    return df[df['Review_Count'] > 0].copy()

def transform_features(df, snapshot_date='2026-01-05'):
    """
    Performs feature transformations and log1p scaling on shortlist features:
    - game_age_days (and log1p version)
    - Price (and log1p version)
    - is_free (binary flag)
    - num_tags (count of tags)
    - primary_genre (first genre in list)
    - Achievements (log1p version)
    - DLC count (log1p version)
    """
    df = df.copy()
    
    # 1. game_age_days
    snapshot = pd.Timestamp(snapshot_date)
    df['game_age_days'] = (snapshot - df['Release date']).dt.days
    df['log_game_age_days'] = np.log1p(df['game_age_days'])
    
    # 2. Price and is_free
    df['log_price'] = np.log1p(df['Price'])
    df['is_free'] = (df['Price'] == 0).astype(int)
    
    # 3. num_tags
    tags_clean = df['Tags'].fillna('')
    df['num_tags'] = tags_clean.apply(lambda x: len(x.split(',')) if x else 0)
    
    # 4. primary_genre
    genres_clean = df['Genres'].fillna('')
    df['primary_genre'] = genres_clean.apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
    
    # 5. Achievements
    df['log_achievements'] = np.log1p(df['Achievements'])
    
    # 6. DLC count
    df['log_dlc_count'] = np.log1p(df['DLC count'])
    
    return df

def encode_genres(df, train_df=None, top_genres=None, top_n=20):
    """
    One-hot encodes the primary_genre feature.
    Ensures that the dummy columns are perfectly aligned with the training set
    to avoid shape/feature mismatches during modeling.
    """
    df = df.copy()
    
    # Determine the top genres to use
    if top_genres is not None:
        pass
    elif train_df is not None:
        # Derive top_genres from the raw primary_genre column of train_df
        top_genres = train_df['primary_genre'].value_counts().nlargest(top_n).index.tolist()
    else:
        # Derive from the current dataframe
        top_genres = df['primary_genre'].value_counts().nlargest(top_n).index.tolist()
        
    df['primary_genre_mapped'] = df['primary_genre'].apply(lambda g: g if g in top_genres else 'Other')
    
    # Enforce categorical type with fixed categories to align dummy columns automatically
    categories = top_genres + ['Other']
    categories = list(dict.fromkeys(categories))  # Remove duplicates if any
    
    df['primary_genre_mapped'] = pd.Categorical(df['primary_genre_mapped'], categories=categories)
    
    # Generate dummy columns
    genre_dummies = pd.get_dummies(df['primary_genre_mapped'], prefix='genre')
    for col in genre_dummies.columns:
        genre_dummies[col] = genre_dummies[col].astype(int)
        
    # Drop any pre-existing genre_ or primary_genre_mapped columns in df to avoid duplication
    pre_existing_cols = [c for c in df.columns if c.startswith('genre_') or c == 'primary_genre_mapped']
    df = df.drop(columns=pre_existing_cols, errors='ignore')
    
    df = pd.concat([df, genre_dummies], axis=1)
    return df, top_genres


def split_data(df):
    """
    Splits the dataset temporally:
    - Train: releases before 2020 (< 2020-01-01)
    - Test: releases from 2020 to 2023 (>= 2020-01-01 and < 2024-01-01)
    - Validation/Out-of-time (OOT): releases from 2024 onward (>= 2024-01-01)
    """
    train_mask = df['Release date'] < '2020-01-01'
    test_mask = (df['Release date'] >= '2020-01-01') & (df['Release date'] < '2024-01-01')
    val_mask = df['Release date'] >= '2024-01-01'
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    val_df = df[val_mask].copy()
    
    return train_df, test_df, val_df
