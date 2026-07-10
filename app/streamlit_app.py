import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Page config
st.set_page_config(
    page_title="Steam Success Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download VADER lexicon silently
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Resolve paths dynamically
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
model_path = project_root / "models" / "best_model.joblib"
features_path = project_root / "models" / "feature_names.joblib"
tag_freq_path = project_root / "models" / "tag_freq_map.joblib"

# Load models and metadata at startup
@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    tag_freq_map = joblib.load(tag_freq_path)
    
    # Apply XGBoost base_score patch to SHAP TreeExplainer
    import shap.explainers._tree
    original_decode = shap.explainers._tree.decode_ubjson_buffer
    
    def patched_decode(fd):
        jmodel = original_decode(fd)
        try:
            param = jmodel["learner"]["learner_model_param"]
            base_score = param["base_score"]
            if isinstance(base_score, str) and base_score.startswith('[') and base_score.endswith(']'):
                param["base_score"] = base_score.strip('[]')
        except Exception:
            pass
        return jmodel
        
    shap.explainers._tree.decode_ubjson_buffer = patched_decode
    explainer = shap.TreeExplainer(model)
    
    return model, features, tag_freq_map, explainer

try:
    model, features, tag_freq_map, explainer = load_resources()
    resources_loaded = True
except Exception as e:
    st.error(f"Error loading model resources: {e}")
    resources_loaded = False

# Custom CSS for rich, premium dark theme styling
st.markdown("""
<style>
    /* Main layout and background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header styling */
    .main-title {
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Card container */
    .custom-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metrics container */
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #58a6ff;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #8b949e;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Advice items */
    .advice-card {
        background-color: #0f141c;
        border-left: 4px solid #58a6ff;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 0 8px 8px 0;
    }
    .advice-title {
        font-weight: bold;
        color: #f0f6fc;
        margin-bottom: 0.2rem;
    }
    .advice-text {
        font-size: 0.9rem;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<h1 class="main-title">🎮 Steam Success Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">An End-to-End Supervised Learning App with Temporal Validation and SHAP Interpretability</p>', unsafe_allow_html=True)

if resources_loaded:
    # Sidebar - Game Metadata Inputs
    st.sidebar.markdown("### 🛠️ Game Launch Parameters")
    
    name = st.sidebar.text_input("Game Title", value="Project Aegis")
    
    primary_genre = st.sidebar.selectbox(
        "Primary Genre",
        options=['Action', 'Adventure', 'Casual', 'Indie', 'Simulation', 'Strategy', 'RPG', 'Violent', 
                 'Free To Play', 'Racing', 'Sexual Content', 'Sports', 'Education', 'Massively Multiplayer', 
                 'Nudity', 'Unknown', 'Gore', 'Utilities', 'Early Access', 'Design & Illustration']
    )
    
    price = st.sidebar.slider("Launch Price ($)", min_value=0.0, max_value=99.99, value=14.99, step=0.49)
    
    desc = st.sidebar.text_area(
        "About The Game (Store Description)",
        value="Embark on an epic adventure through post-apocalyptic worlds. Fight challenging enemies, discover hidden secrets, and build your base in this action-packed RPG. Deep customization, cooperative multiplayer, and beautiful graphics.",
        height=120
    )
    
    achievements = st.sidebar.slider("Number of Achievements", min_value=0, max_value=1000, value=50, step=5)
    dlc_count = st.sidebar.slider("Number of Planned DLCs", min_value=0, max_value=20, value=0, step=1)
    
    st.sidebar.markdown("#### Platforms")
    win = st.sidebar.checkbox("Windows", value=True)
    mac = st.sidebar.checkbox("Mac OS", value=False)
    linux = st.sidebar.checkbox("Linux", value=False)
    
    age = st.sidebar.selectbox("Age Rating Required", options=[0, 12, 16, 18], index=0)
    release_year = st.sidebar.slider("Release Year", min_value=2024, max_value=2030, value=2026)
    
    # Sidebar Tags selection (top 30 tags by frequency for easy entry)
    st.sidebar.markdown("#### Tags Selection")
    sorted_tags = sorted(tag_freq_map.items(), key=lambda x: x[1], reverse=True)
    top_30_tags = [tag for tag, freq in sorted_tags[:30]]
    selected_tags = st.sidebar.multiselect("Select Tags (min 1 recommended)", options=top_30_tags, default=["Indie", "Action", "RPG"])
    
    # 2-column Main Layout
    col1, col2 = st.columns([1, 1])
    
    # Column 1: Feature Processing & Diagnostics
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📋 Parsed Game Features")
        
        # Calculate feature variables
        is_free = 1 if price == 0 else 0
        log_price = np.log1p(price)
        num_tags = len(selected_tags)
        
        # Format tags string
        tags_str = ",".join(selected_tags)
        
        # Compute tag frequencies
        tag_freqs = [tag_freq_map.get(t, 0) for t in selected_tags]
        tag_freq_mean = np.mean(tag_freqs) if tag_freqs else 0.0
        tag_freq_max = np.max(tag_freqs) if tag_freqs else 0.0
        tag_freq_min = np.min(tag_freqs) if tag_freqs else 0.0
        tag_freq_sum = np.sum(tag_freqs) if tag_freqs else 0.0
        
        # Compute VADER sentiment
        sia = SentimentIntensityAnalyzer()
        desc_sentiment = sia.polarity_scores(desc)['compound']
        
        # Build features dataframe
        input_data = {
            'Required age': age,
            'Price': price,
            'Discount': 0.0,
            'DLC count': dlc_count,
            'Windows': 1 if win else 0,
            'Mac': 1 if mac else 0,
            'Linux': 1 if linux else 0,
            'Metacritic score': 0.0,
            'User score': 0.0,
            'Achievements': achievements,
            'Average playtime forever': 0.0,
            'Average playtime two weeks': 0.0,
            'Median playtime forever': 0.0,
            'Median playtime two weeks': 0.0,
            'game_age_days': 0.0, # assumed launch day (age 0)
            'log_game_age_days': 0.0,
            'log_price': log_price,
            'is_free': is_free,
            'num_tags': num_tags,
            'log_achievements': np.log1p(achievements),
            'log_dlc_count': np.log1p(dlc_count),
            'release_year': release_year,
            'num_genres': len(primary_genre.split(',')),
            'tag_freq_mean': tag_freq_mean,
            'tag_freq_max': tag_freq_max,
            'tag_freq_min': tag_freq_min,
            'tag_freq_sum': tag_freq_sum,
            'desc_sentiment_score': desc_sentiment
        }
        
        # Price Tiers
        input_data['price_tier_free'] = 1 if price == 0 else 0
        input_data['price_tier_budget'] = 1 if 0 < price <= 5.0 else 0
        input_data['price_tier_mid'] = 1 if 5.0 < price <= 20.0 else 0
        input_data['price_tier_premium'] = 1 if price > 20.0 else 0
        
        # Genres One-hot
        top_genres = ['Action', 'Adventure', 'Casual', 'Indie', 'Simulation', 'Strategy', 'RPG', 'Violent', 
                      'Free To Play', 'Racing', 'Sexual Content', 'Sports', 'Education', 'Massively Multiplayer', 
                      'Nudity', 'Unknown', 'Gore', 'Utilities', 'Early Access', 'Design & Illustration']
                      
        for g in top_genres:
            input_data[f'genre_{g}'] = 1 if primary_genre == g else 0
        input_data['genre_Other'] = 1 if primary_genre not in top_genres else 0
        
        # Match feature order exactly
        X_input = pd.DataFrame([input_data])[features]
        
        # Render feature summary
        feat_df = pd.DataFrame({
            'Feature': ['Primary Genre', 'Launch Price', 'Selected Tags Count', 'VADER Description Sentiment', 'Achievements Count', 'Windows OS Target'],
            'Parsed Value': [str(primary_genre), f"${price}", str(num_tags), f"{desc_sentiment:.2f}", str(achievements), "Yes" if win else "No"]
        })
        st.table(feat_df)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Actionable Optimization Panel
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("💡 Optimization Recommendations")
        
        # Predict score
        pred_val = model.predict(X_input)[0]
        
        # Generate SHAP value attributions
        shap_values_input = explainer(X_input)
        shap_attribs = dict(zip(features, shap_values_input.values[0]))
        
        recommendations = []
        
        # 1. Tags check
        if num_tags < 8:
            recommendations.append({
                "title": "Increase Discoverability (Tags)",
                "text": f"Your game only has {num_tags} tags. Adding more tags (aim for 10-15) helps Steam's recommendation algorithm catalog and distribute your game."
            })
            
        # 2. Price check
        if shap_attribs.get('log_price', 0) < -0.1:
            recommendations.append({
                "title": "Optimize Pricing",
                "text": f"The launch price of ${price} is pulling down the predicted score. Consider benchmarking against competitors in your genre."
            })
            
        # 3. Description Sentiment Check
        if desc_sentiment < 0.3:
            recommendations.append({
                "title": "Boost Description Sentiment",
                "text": f"Your current description sentiment score is {desc_sentiment:.2f}. Try adding more positive descriptors, exciting gameplay verbs, or feature lists to boost sentiment."
            })
            
        # 4. OS check
        if not win:
            recommendations.append({
                "title": "Critical Platform Missing",
                "text": "Your game does not target Windows. Windows accounts for over 95% of active gamers on Steam. Adding Windows support is essential."
            })
            
        if not recommendations:
            recommendations.append({
                "title": "Excellent Baseline Profile!",
                "text": "Your game launch profile has strong baseline markers across platform coverage, tags density, description sentiment, and pricing."
            })
            
        for rec in recommendations:
            st.markdown(f"""
            <div class="advice-card">
                <div class="advice-title">{rec['title']}</div>
                <div class="advice-text">{rec['text']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Column 2: Predictions & Local SHAP waterfall
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Predicted Success Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{pred_val:.2f}</div>', unsafe_allow_html=True)
        
        # Contextual rating
        if pred_val < 2.0:
            status = "Low Engagement (typical of niche/asset-heavy titles)"
            color = "gray"
        elif pred_val < 4.5:
            status = "Moderate Engagement (typical of standard budget indies)"
            color = "orange"
        elif pred_val < 7.0:
            status = "High Engagement (strong commercial potential)"
            color = "green"
        else:
            status = "Breakout Hit Potential (comparable to top Steam titles)"
            color = "blue"
            
        st.markdown(f"<p style='text-align:center; font-weight:bold; color:{color};'>{status}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🎯 Feature Contribution (SHAP Attribution)")
        
        # Render SHAP waterfall plot
        fig, ax = plt.subplots(figsize=(8, 5))
        # Ensure plot has dark background match
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        plt.rcParams['text.color'] = '#c9d1d9'
        plt.rcParams['axes.labelcolor'] = '#c9d1d9'
        plt.rcParams['xtick.color'] = '#c9d1d9'
        plt.rcParams['ytick.color'] = '#c9d1d9'
        
        shap.plots.waterfall(shap_values_input[0], max_display=10, show=False)
        plt.title("How Launch Metadata Influenced Prediction", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Model and resources are missing. Run the data and training pipelines first to serialize the model artifacts.")
