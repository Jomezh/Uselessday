"""
Chaos Analyzer Pro — Camera input that resets after analysis
Dependencies:
  pip install streamlit ultralytics opencv-python scikit-image numpy pillow
Run:
  streamlit run app.py
"""

import time, math
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from skimage.measure import shannon_entropy
from ultralytics import YOLO

st.set_page_config(page_title="Chaos Analyzer Pro", page_icon="🌪️", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0
if 'clear_camera' not in st.session_state:
    st.session_state.clear_camera = False

def preprocess_image(img, target_width=640, mirror=False):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if len(img.shape) == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = img
    
    h, w = bgr.shape[:2]
    new_h = int(h * (target_width / w))
    resized = cv2.resize(bgr, (target_width, new_h))
    
    if mirror:
        resized = cv2.flip(resized, 1)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    return resized, gray, blurred

def compute_features(gray, canny_low, canny_high, window_size, stride):
    # Edge density
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Entropy
    entropy = shannon_entropy(gray)
    
    # Local chaos
    h, w = edges.shape
    window_eds = []
    rows = max(1, (h - window_size) // stride + 1)
    cols = max(1, (w - window_size) // stride + 1)
    chaos_map = np.zeros((rows, cols))
    
    i = 0
    for y in range(0, max(1, h - window_size + 1), stride):
        j = 0
        for x in range(0, max(1, w - window_size + 1), stride):
            window = edges[y:y+window_size, x:x+window_size]
            window_ed = np.count_nonzero(window) / window.size if window.size > 0 else 0
            window_eds.append(window_ed)
            if i < rows and j < cols:
                chaos_map[i, j] = window_ed
            j += 1
        i += 1
    
    local_chaos = np.std(window_eds) if window_eds else 0
    
    return {
        "edges": edges,
        "edge_density": edge_density,
        "entropy": entropy, 
        "local_chaos": local_chaos,
        "chaos_map": chaos_map
    }

def detect_objects(bgr, model, conf_thresh):
    results = model.predict(bgr, imgsz=640, conf=conf_thresh, verbose=False)
    detections = []
    
    if len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                name = r.names[cls]
                detections.append((x1, y1, x2, y2, conf, name))
    
    # Object features
    count = len(detections)
    h, w = bgr.shape[:2]
    area = h * w
    obj_density = count / area
    
    # Weighted messiness
    weights = {"cup": 1.0, "bottle": 1.0, "book": 1.0, "chair": 0.6, "laptop": 0.4, "tv": 0.4}
    weighted_mess = sum(weights.get(name, 0.5) for _, _, _, _, _, name in detections) / area
    
    # Dispersion
    if count >= 2:
        centroids = [[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2,_,_ in detections]
        centroids = np.array(centroids)
        var = np.var(centroids, axis=0).mean()
        diag = math.sqrt(w*w + h*h)
        dispersion = var / (diag * diag)
    else:
        dispersion = 0
    
    return detections, {
        "object_count": count,
        "object_density": obj_density, 
        "weighted_mess": weighted_mess,
        "dispersion": dispersion
    }

def draw_detections(bgr, detections):
    img = bgr.copy()
    for x1, y1, x2, y2, conf, name in detections:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{name}: {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def compute_chaos_score(low_features, obj_features, bounds, weights, coffee_mode):
    # Normalize features
    def normalize(x, min_val, max_val):
        return max(0, min(100, (x - min_val) / (max_val - min_val) * 100))
    
    ed_norm = normalize(low_features["edge_density"], bounds["ed_min"], bounds["ed_max"])
    h_norm = normalize(low_features["entropy"], bounds["h_min"], bounds["h_max"])  
    lc_norm = normalize(low_features["local_chaos"], bounds["lc_min"], bounds["lc_max"])
    od_norm = normalize(obj_features["object_density"], 0, bounds["od_max"])
    wm_norm = normalize(obj_features["weighted_mess"], 0, bounds["wm_max"])
    dp_norm = normalize(obj_features["dispersion"], 0, bounds["dp_max"])
    
    # Calculate weighted score
    score = (weights["edge"] * ed_norm + 
            weights["entropy"] * h_norm +
            weights["local_chaos"] * lc_norm +
            weights["obj_density"] * od_norm +
            weights["weighted_mess"] * wm_norm +
            weights["dispersion"] * dp_norm) * coffee_mode
    
    score = max(0, min(100, score))
    
    return score, {
        "ed_norm": ed_norm, "h_norm": h_norm, "lc_norm": lc_norm,
        "od_norm": od_norm, "wm_norm": wm_norm, "dp_norm": dp_norm
    }

def get_chaos_label(score):
    if score <= 15: return "🧘 Monk Mode"
    elif score <= 35: return "✨ Neat Nook"  
    elif score <= 55: return "🎯 Controlled Chaos"
    elif score <= 75: return "🌪️ Hurricane Hover"
    else: return "👹 Goblin Lair"

def get_roast(score, detections):
    cups = sum(1 for *_, name in detections if "cup" in name or "bottle" in name)
    books = sum(1 for *_, name in detections if "book" in name)
    
    if score < 20:
        return "✨ Minimalism called; it's proud of you."
    elif score < 50:
        if books >= 3:
            return "📚 A library in progress—Dewey would approve."
        return "🤷 Somewhere between zen and 'I'll deal with it tomorrow.'"
    elif score < 80:
        if cups >= 5:
            return "☕ Hydration station detected—this doubles as recycling."
        return "📦 Your floor is Schrödinger's desk."
    else:
        return "🔬 Anthropologists would love to study this habitat."

def create_progress_bar(score):
    if score <= 25: color = "#4CAF50"
    elif score <= 50: color = "#FF9800"  
    elif score <= 75: color = "#F44336"
    else: color = "#9C27B0"
    
    return f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
        <div style="width: {score:.1f}%; background-color: {color}; height: 30px; 
                    display: flex; align-items: center; justify-content: center; 
                    color: white; font-weight: bold; transition: width 0.3s ease;">
            {score:.1f}%
        </div>
    </div>
    """

# Load model
model = load_model()

# UI Layout
st.title("🌪️ Chaos Analyzer Pro")
st.caption("Take photos, analyze chaos, then take new photos without freezing!")

# Sidebar controls  
with st.sidebar:
    st.header("🎛️ Controls")
    
    input_mode = st.radio("Input Mode", ["📷 Camera", "📁 Upload"], index=0)
    
    if input_mode == "📷 Camera":
        mirror_cam = st.checkbox("🪞 Mirror camera", value=True)
        auto_analyze = st.checkbox("🔄 Auto analyze", value=False)
        if auto_analyze:
            auto_interval = st.slider("Auto interval (seconds)", 0.5, 3.0, 1.0, 0.1)
    
    st.subheader("🎯 Analysis Parameters")
    canny_low = st.slider("Canny Low Threshold", 10, 150, 50)
    canny_high = st.slider("Canny High Threshold", 100, 300, 150)
    window_size = st.slider("Local Chaos Window", 32, 128, 64, 16)
    stride = st.slider("Window Stride", 16, 64, 32, 8)
    conf_thresh = st.slider("Object Detection Confidence", 0.1, 0.9, 0.35, 0.05)
    
    st.subheader("⚖️ Feature Weights")
    weights = {
        "edge": st.slider("Edge Density", 0.0, 1.0, 0.35, 0.05),
        "entropy": st.slider("Entropy", 0.0, 1.0, 0.20, 0.05),
        "local_chaos": st.slider("Local Chaos", 0.0, 1.0, 0.10, 0.05),
        "obj_density": st.slider("Object Density", 0.0, 1.0, 0.20, 0.05),
        "weighted_mess": st.slider("Weighted Messiness", 0.0, 1.0, 0.10, 0.05),
        "dispersion": st.slider("Object Dispersion", 0.0, 1.0, 0.05, 0.05)
    }
    
    st.subheader("📊 Normalization Bounds")
    bounds = {
        "ed_min": 0.02, "ed_max": 0.15,
        "h_min": 4.0, "h_max": 7.5, 
        "lc_min": 0.01, "lc_max": 0.06,
        "od_max": st.number_input("Object Density Max", value=0.002, format="%.6f"),
        "wm_max": st.number_input("Weighted Mess Max", value=0.004, format="%.6f"),
        "dp_max": st.number_input("Dispersion Max", value=0.06, format="%.3f")
    }
    
    st.subheader("☕ Judging Mode")
    coffee_mode = st.selectbox("Mood", 
        options=[0.9, 1.0, 1.1],
        index=1,
        format_func=lambda x: "After Coffee (Gentle)" if x==0.9 else "Normal" if x==1.0 else "Before Coffee (Harsh)"
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Input")
    
    current_image = None
    
    if input_mode == "📷 Camera":
        st.info("📷 Take a photo to analyze. Camera resets after each analysis for new photos!")
        
        # Key trick: use a key that changes to reset the camera input
        camera_key = f"camera_input_{st.session_state.get('photo_counter', 0)}"
        
        # Use Streamlit's built-in camera input with dynamic key
        camera_photo = st.camera_input("Take a picture", key=camera_key)
        
        if camera_photo is not None:
            current_image = Image.open(camera_photo)
            
            # Show current photo
            st.image(current_image, caption="Current Photo", use_container_width=True)
            
            # Analysis controls
            col_analyze, col_reset = st.columns([2, 1])
            
            with col_analyze:
                analyze_now = st.button("🔍 Analyze This Photo", type="primary", use_container_width=True)
            
            with col_reset:
                reset_camera = st.button("🔄 New Photo", use_container_width=True)
            
            # Auto analyze logic
            current_time = time.time()
            should_auto_analyze = (auto_analyze and 
                                 current_time - st.session_state.last_analysis_time > auto_interval)
            
            if analyze_now or should_auto_analyze:
                with st.spinner("Analyzing chaos..."):
                    # Process image
                    bgr, gray, _ = preprocess_image(current_image, mirror=mirror_cam)
                    
                    # Compute features
                    low_feat = compute_features(gray, canny_low, canny_high, window_size, stride)
                    detections, obj_feat = detect_objects(bgr, model, conf_thresh)
                    
                    # Calculate score
                    score, norms = compute_chaos_score(low_feat, obj_feat, bounds, weights, coffee_mode)
                    
                    # Store results
                    st.session_state.analysis_result = {
                        "score": score,
                        "label": get_chaos_label(score),
                        "roast": get_roast(score, detections),
                        "norms": norms,
                        "detections": detections,
                        "low_features": low_feat,
                        "obj_features": obj_feat,
                        "annotated_image": draw_detections(bgr, detections)
                    }
                    st.session_state.last_analysis_time = current_time
                    
                st.success("✅ Analysis complete! Take a new photo for another analysis.")
            
            # Reset camera to take new photo
            if reset_camera or analyze_now:
                # Increment counter to change the key and reset camera input
                st.session_state.photo_counter = st.session_state.get('photo_counter', 0) + 1
                st.rerun()
    
    else:  # Upload mode
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            current_image = Image.open(uploaded_file)
            st.image(current_image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Uploaded Image", type="primary"):
                with st.spinner("Analyzing chaos..."):
                    # Process image  
                    bgr, gray, _ = preprocess_image(current_image)
                    
                    # Compute features
                    low_feat = compute_features(gray, canny_low, canny_high, window_size, stride)
                    detections, obj_feat = detect_objects(bgr, model, conf_thresh)
                    
                    # Calculate score
                    score, norms = compute_chaos_score(low_feat, obj_feat, bounds, weights, coffee_mode)
                    
                    # Store results
                    st.session_state.analysis_result = {
                        "score": score,
                        "label": get_chaos_label(score), 
                        "roast": get_roast(score, detections),
                        "norms": norms,
                        "detections": detections,
                        "low_features": low_feat,
                        "obj_features": obj_feat,
                        "annotated_image": draw_detections(bgr, detections)
                    }
                    
                st.success("✅ Analysis complete!")

with col2:
    st.subheader("📊 Analysis Results")
    
    # Display results if available
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Main score display
        st.markdown(f"### {result['label']}")
        st.markdown(create_progress_bar(result["score"]), unsafe_allow_html=True)
        
        # Feature breakdown
        st.markdown("**Feature Breakdown:**")
        norms = result["norms"]
        st.write(f"• Edge Density: {norms['ed_norm']:.1f}/100")
        st.write(f"• Entropy: {norms['h_norm']:.1f}/100") 
        st.write(f"• Local Chaos: {norms['lc_norm']:.1f}/100")
        st.write(f"• Object Density: {norms['od_norm']:.1f}/100")
        st.write(f"• Weighted Messiness: {norms['wm_norm']:.1f}/100")
        st.write(f"• Dispersion: {norms['dp_norm']:.1f}/100")
        
        st.write(f"**Objects Detected:** {result['obj_features']['object_count']}")
        
        # Roast
        if st.button("🔥 Roast My Room!"):
            st.markdown(f"### 🔥 {result['roast']}")
        
        # Annotated image
        st.image(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB), 
                caption="Detected Objects", use_container_width=True)
        
        # Optional overlays
        show_overlays = st.checkbox("📈 Show Analysis Overlays")
        if show_overlays:
            overlay_col1, overlay_col2 = st.columns(2)
            
            with overlay_col1:
                st.image(result["low_features"]["edges"], 
                        caption="Edge Map", clamp=True, use_container_width=True)
            
            with overlay_col2:
                if result["low_features"]["chaos_map"].size > 0:
                    chaos_map = result["low_features"]["chaos_map"]
                    # Normalize and colorize
                    normalized = (chaos_map - chaos_map.min()) / (chaos_map.max() - chaos_map.min() + 1e-8)
                    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    st.image(colored, caption="Local Chaos Heatmap", use_container_width=True)
        
    else:
        st.info("👆 Take a photo and click Analyze to see results!")

# Footer
st.markdown("---")
st.caption("🎯 Camera automatically resets after analysis for continuous photo-taking • No freezing or WebRTC complexity")
