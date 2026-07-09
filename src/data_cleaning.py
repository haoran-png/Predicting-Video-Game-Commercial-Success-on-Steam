import pandas as pd
import numpy as np
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

# Advanced Feature Engineering Functions (Combined from Notebook 5)

def add_pricing_tiers(df):
    """
    Categorizes Price into four tiers: free, budget, mid, premium.
    One-hot encodes the price tier, guaranteeing identical shapes.
    """
    df = df.copy()
    
    def get_tier(price):
        if price == 0:
            return 'free'
        elif price <= 5.0:
            return 'budget'
        elif price <= 20.0:
            return 'mid'
        else:
            return 'premium'
            
    df['price_tier'] = df['Price'].apply(get_tier)
    
    # Enforce categorical type to align dummy columns automatically
    categories = ['free', 'budget', 'mid', 'premium']
    df['price_tier'] = pd.Categorical(df['price_tier'], categories=categories)
    
    price_dummies = pd.get_dummies(df['price_tier'], prefix='price_tier')
    for col in price_dummies.columns:
        price_dummies[col] = price_dummies[col].astype(int)
        
    # Drop intermediate columns if any
    pre_existing = [c for c in df.columns if c.startswith('price_tier_')]
    df = df.drop(columns=pre_existing, errors='ignore')
    
    df = pd.concat([df, price_dummies], axis=1)
    return df

def add_genre_count(df):
    """
    Counts the number of genres a game belongs to.
    """
    df = df.copy()
    genres_clean = df['Genres'].fillna('')
    df['num_genres'] = genres_clean.apply(lambda x: len([g.strip() for g in x.split(',') if g.strip()]) if x else 0)
    return df

def fit_tag_frequencies(train_df):
    """
    Fits tag frequency mapping based ONLY on training data to avoid future leakage.
    """
    tags_series = train_df['Tags'].fillna('')
    tag_counter = Counter()
    for tags_str in tags_series:
        if tags_str:
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            tag_counter.update(tags)
    return dict(tag_counter)

def add_tag_frequency_features(df, tag_freq_map):
    """
    Maps tags to training set frequencies and computes aggregates: mean, max, min, sum.
    """
    df = df.copy()
    
    def get_tag_stats(tags_str):
        if not tags_str or pd.isna(tags_str):
            return 0.0, 0.0, 0.0, 0.0
        tags = [t.strip() for t in tags_str.split(',') if t.strip()]
        if not tags:
            return 0.0, 0.0, 0.0, 0.0
        freqs = [tag_freq_map.get(t, 0) for t in tags]
        return float(np.mean(freqs)), float(np.max(freqs)), float(np.min(freqs)), float(np.sum(freqs))

    stats = df['Tags'].apply(get_tag_stats)
    
    df['tag_freq_mean'] = stats.apply(lambda x: x[0])
    df['tag_freq_max'] = stats.apply(lambda x: x[1])
    df['tag_freq_min'] = stats.apply(lambda x: x[2])
    df['tag_freq_sum'] = stats.apply(lambda x: x[3])
    
    return df

def add_description_sentiment(df):
    """
    Calculates the compound sentiment score from NLTK VADER on the description column.
    """
    df = df.copy()
    sia = SentimentIntensityAnalyzer()
    
    # Handle missing values
    descriptions = df['About the game'].fillna('')
    
    # Compute scores using list comprehension (fastest execution path)
    df['desc_sentiment_score'] = [sia.polarity_scores(text)['compound'] for text in descriptions]
    return df

def add_temporal_features(df):
    """
    Extracts release_year from the 'Release date' column.
    """
    df = df.copy()
    if 'Release date' in df.columns:
        df['release_year'] = pd.to_datetime(df['Release date']).dt.year
    return df
