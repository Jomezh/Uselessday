"""
Chaos Analyzer - A fun hackathon project to analyze room chaos
Dependencies: pip install streamlit opencv-python scikit-image numpy pillow
Run with: streamlit run chaos_analyzer.py
"""

import streamlit as st
import cv2
import numpy as np
from skimage.measure import shannon_entropy
from PIL import Image
import io

# App configuration
st.set_page_config(
    page_title="🌪️ Chaos Analyzer",
    page_icon="🌪️",
    layout="wide"
)

def preprocess_image(image, target_width=640):
    """Resize and preprocess image for analysis"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize maintaining aspect ratio
    height, width = image.shape[:2]
    new_width = target_width
    new_height = int(height * target_width / width)
    resized = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return resized, gray, blurred

def compute_features(gray_img, canny_low=50, canny_high=150, window_size=64, stride=32):
    """Compute chaos features: edge density, entropy, local chaos"""
    
    # Edge density
    edges = cv2.Canny(gray_img, canny_low, canny_high)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Shannon entropy
    entropy = shannon_entropy(gray_img)
    
    # Local chaos (variance of edge density in windows)
    h, w = edges.shape
    window_eds = []
    local_chaos_map = np.zeros((h//stride + 1, w//stride + 1))
    
    for i, y in enumerate(range(0, h - window_size + 1, stride)):
        for j, x in enumerate(range(0, w - window_size + 1, stride)):
            window = edges[y:y+window_size, x:x+window_size]
            window_ed = np.count_nonzero(window) / window.size
            window_eds.append(window_ed)
            local_chaos_map[i, j] = window_ed
    
    local_chaos = np.std(window_eds) if window_eds else 0
    
    return {
        'edge_density': edge_density,
        'entropy': entropy,
        'local_chaos': local_chaos,
        'edges': edges,
        'local_chaos_map': local_chaos_map
    }

def normalize_and_score(features, coffee_mode=1.0):
    """Normalize features and compute final chaos score"""
    
    # Normalization bounds (tunable)
    ed_min, ed_max = 0.02, 0.15
    h_min, h_max = 4.0, 7.5
    lc_min, lc_max = 0.01, 0.06
    
    # Normalize to 0-100
    ed_norm = np.clip((features['edge_density'] - ed_min) / (ed_max - ed_min), 0, 1) * 100
    h_norm = np.clip((features['entropy'] - h_min) / (h_max - h_min), 0, 1) * 100
    lc_norm = np.clip((features['local_chaos'] - lc_min) / (lc_max - lc_min), 0, 1) * 100
    
    # Weighted final score
    chaos_score = (0.5 * ed_norm + 0.35 * h_norm + 0.15 * lc_norm) * coffee_mode
    chaos_score = np.clip(chaos_score, 0, 100)
    
    return {
        'chaos_score': chaos_score,
        'ed_norm': ed_norm,
        'h_norm': h_norm,
        'lc_norm': lc_norm
    }

def get_chaos_label(score):
    """Get fun label based on chaos score"""
    if score <= 15:
        return "🧘 Monk Mode"
    elif score <= 35:
        return "✨ Neat Nook"
    elif score <= 55:
        return "🎯 Controlled Chaos"
    elif score <= 75:
        return "🌪️ Hurricane Hover"
    else:
        return "👹 Goblin Lair"

def get_roast(score):
    """Get playful roast based on score"""
    if score < 20:
        return "Minimalism called; it's proud of you. ✨"
    elif score < 50:
        return "Somewhere between zen and 'I'll deal with it tomorrow.' 🤷"
    elif score < 80:
        return "Your floor is Schrödinger's desk. 📦"
    else:
        return "Anthropologists would like to study this habitat. 🔬"

def render_chaos_meter(score):
    """Render a visual chaos meter"""
    # Color based on score
    if score <= 25:
        color = "#00ff00"  # Green
    elif score <= 50:
        color = "#ffff00"  # Yellow
    elif score <= 75:
        color = "#ff8000"  # Orange
    else:
        color = "#ff0000"  # Red
    
    # Create progress bar HTML
    progress_html = f"""
    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px; margin: 10px 0;">
        <div style="background-color: {color}; width: {score}%; height: 30px; border-radius: 7px; 
                    display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
            {score:.1f}%
        </div>
    </div>
    """
    return progress_html

# Main app
st.title("🌪️ Chaos Analyzer")
st.markdown("*Analyze the chaos in your room with scientific precision and playful roasts!*")

# Sidebar controls
st.sidebar.header("🎛️ Controls")

# Image input method
input_method = st.sidebar.radio("Input Method", ["📁 Upload Image", "📷 Use Camera"])

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
canny_low = st.sidebar.slider("Canny Low Threshold", 10, 100, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 100, 300, 150)
window_size = st.sidebar.slider("Window Size", 32, 128, 64, step=16)

# Coffee mode
coffee_mode = st.sidebar.selectbox(
    "☕ Judging Mode",
    options=[0.9, 1.0, 1.1],
    index=1,
    format_func=lambda x: "Before Coffee (Harsh)" if x == 1.1 else "Normal" if x == 1.0 else "After Coffee (Gentle)"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Input")
    
    image = None
    if input_method == "📁 Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    elif input_method == "📷 Use Camera":
        camera_input = st.camera_input("Take a picture")
        if camera_input is not None:
            image = Image.open(camera_input)
            st.image(image, caption="Camera Input", use_column_width=True)

with col2:
    st.header("📊 Analysis Results")
    
    if image is not None:
        # Process image
        resized, gray, blurred = preprocess_image(image)
        features = compute_features(gray, canny_low, canny_high, window_size)
        scores = normalize_and_score(features, coffee_mode)
        
        # Display main results
        chaos_score = scores['chaos_score']
        label = get_chaos_label(chaos_score)
        
        st.markdown(f"### {label}")
        st.markdown(render_chaos_meter(chaos_score), unsafe_allow_html=True)
        
        # Roast button
        if st.button("🔥 Roast My Room"):
            roast = get_roast(chaos_score)
            st.markdown(f"**{roast}**")
        
        # Feature breakdown
        with st.expander("📈 Feature Breakdown"):
            st.write(f"**Edge Density:** {scores['ed_norm']:.1f}/100")
            st.write(f"**Entropy:** {scores['h_norm']:.1f}/100")
            st.write(f"**Local Chaos:** {scores['lc_norm']:.1f}/100")
            st.write(f"**Raw Edge Density:** {features['edge_density']:.4f}")
            st.write(f"**Raw Entropy:** {features['entropy']:.2f}")
            st.write(f"**Raw Local Chaos:** {features['local_chaos']:.4f}")

# Additional visualizations
if image is not None:
    st.header("🔍 Visual Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if st.checkbox("Show Edge Map"):
            st.image(features['edges'], caption="Edge Detection", use_column_width=True, clamp=True)
    
    with viz_col2:
        if st.checkbox("Show Local Chaos Heatmap"):
            # Create heatmap visualization
            chaos_map = features['local_chaos_map']
            if chaos_map.size > 0:
                # Normalize and colorize
                normalized_map = (chaos_map - chaos_map.min()) / (chaos_map.max() - chaos_map.min() + 1e-8)
                colored_map = cv2.applyColorMap((normalized_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                st.image(colored_map, caption="Local Chaos Heatmap (Red = Chaotic)", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for hackathons. Made with Streamlit, OpenCV, and questionable life choices.*")
