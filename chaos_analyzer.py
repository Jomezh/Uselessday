"""
Chaos Analyzer Pro - Fixed version with proper error handling
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

st.set_page_config(page_title="🌪️ Chaos Analyzer ", page_icon="🌪️", layout="wide")

@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        st.success("✅ YOLO model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {str(e)}")
        st.info("💡 Try: pip install ultralytics")
        return None

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0
if 'photo_counter' not in st.session_state:
    st.session_state.photo_counter = 0

def mirror_image_if_needed(image, mirror_enabled):
    """Apply mirroring to both PIL and numpy images for consistent display"""
    if not mirror_enabled:
        return image
    
    if isinstance(image, Image.Image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif isinstance(image, np.ndarray):
        return cv2.flip(image, 1)  # Horizontal flip
    else:
        return image

def preprocess_image(img, target_width=640, mirror=False):
    """Fixed preprocessing function"""
    try:
        if isinstance(img, Image.Image):
            if mirror:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = np.array(img)
        
        # FIXED: Proper shape checking
        if len(img.shape) == 3 and img.shape[2] == 3:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            bgr = img
        
        h, w = bgr.shape[:2]
        if w != target_width:
            new_h = int(h * (target_width / w))
            resized = cv2.resize(bgr, (target_width, new_h))
        else:
            resized = bgr
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return resized, gray
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None

def detect_objects(bgr, model, conf_thresh):
    """FIXED: Proper YOLO detection with error handling"""
    if model is None:
        return [], {"object_count": 0, "object_density": 0}
    
    try:
        results = model.predict(bgr, imgsz=640, conf=conf_thresh, verbose=False)
        detections = []
        
        if len(results) > 0:
            r = results[0]  # FIXED: Proper indexing
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    # FIXED: Proper tensor handling
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    name = r.names[cls]
                    detections.append((x1, y1, x2, y2, conf, name))
        
        count = len(detections)
        h, w = bgr.shape[:2]
        area = h * w
        obj_density = count / area if area > 0 else 0
        
        return detections, {"object_count": count, "object_density": obj_density}
    
    except Exception as e:
        st.error(f"YOLO detection error: {e}")
        return [], {"object_count": 0, "object_density": 0}

def draw_detections(bgr, detections):
    """Draw bounding boxes on detected objects"""
    if len(detections) == 0:
        return bgr
    
    img = bgr.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, (x1, y1, x2, y2, conf, name) in enumerate(detections):
        color = colors[i % len(colors)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name}: {conf:.2f}"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (int(x1), int(y1) - text_height - 10), 
                     (int(x1) + text_width, int(y1)), color, -1)
        
        cv2.putText(img, label, (int(x1), int(y1-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def compute_chaos_features(gray):
    """Compute chaos features with proper error handling"""
    try:
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size if edges.size > 0 else 0
        
        # Shannon entropy
        entropy = shannon_entropy(gray)
        
        # Local chaos with heatmap
        h, w = edges.shape
        window_size = 64
        stride = 32
        
        rows = max(1, (h - window_size) // stride + 1)
        cols = max(1, (w - window_size) // stride + 1)
        chaos_map = np.zeros((rows, cols))
        
        window_eds = []
        i = 0
        for y in range(0, h - window_size + 1, stride):
            j = 0
            for x in range(0, w - window_size + 1, stride):
                if i < rows and j < cols:
                    window = edges[y:y+window_size, x:x+window_size]
                    window_ed = np.count_nonzero(window) / window.size if window.size > 0 else 0
                    window_eds.append(window_ed)
                    chaos_map[i, j] = window_ed
                j += 1
            i += 1
        
        local_chaos = np.std(window_eds) if len(window_eds) > 0 else 0
        
        return {
            "edges": edges,
            "edge_density": edge_density,
            "entropy": entropy,
            "local_chaos": local_chaos,
            "chaos_map": chaos_map
        }
    except Exception as e:
        st.error(f"Feature computation error: {e}")
        return None

def compute_chaos_score(features, detections, coffee_mode):
    """Simplified but effective chaos scoring"""
    try:
        if features is None:
            return 0, {}
        
        # Research-calibrated scoring
        edge_score = min(100, features["edge_density"] * 400)  # 0.02-0.15 -> 0-100
        entropy_score = min(100, max(0, (features["entropy"] - 4.0) * 25))  # 4-7.5 -> 0-100  
        local_score = min(100, features["local_chaos"] * 1000)  # 0-0.1 -> 0-100
        object_score = min(100, len(detections) * 12)  # Up to 8 objects = 96 points
        
        # Weighted combination
        final_score = (
            0.30 * edge_score +      # Visual complexity
            0.25 * entropy_score +   # Randomness
            0.25 * local_score +     # Local variation
            0.20 * object_score      # Object clutter
        ) * coffee_mode
        
        component_scores = {
            "edge_density": edge_score,
            "entropy": entropy_score,
            "local_chaos": local_score,
            "object_count": object_score
        }
        
        return max(0, min(100, final_score)), component_scores
    
    except Exception as e:
        st.error(f"Scoring error: {e}")
        return 0, {}

def get_chaos_label(score):
    if score <= 20: return "🧘 Monk Mode"
    elif score <= 35: return "✨ Neat Nook"  
    elif score <= 55: return "🎯 Controlled Chaos"
    elif score <= 75: return "🌪️ Hurricane Hover"
    else: return "👹 Goblin Lair"

def get_roast(score, detections):
    cups_bottles = sum(1 for *_, name in detections if any(x in name.lower() for x in ["cup", "bottle", "glass"]))
    books = sum(1 for *_, name in detections if "book" in name.lower())
    food_items = sum(1 for *_, name in detections if any(x in name.lower() for x in ["pizza", "banana", "apple", "orange", "cake", "donut"]))
    
    if score < 25:
        roasts = [
            "✨ Minimalism called; it's proud of you.",
            "🧘 Marie Kondo would shed a single, perfect tear.",
            "🏛️ This level of organization belongs in a museum."
        ]
    elif score < 50:
        if books >= 3:
            return "📚 A library in progress—Dewey Decimal could help organize this knowledge fortress."
        roasts = [
            "🤷 Somewhere between zen and 'I'll deal with it tomorrow.'",
            "🎯 You're walking the fine line between organized and 'creative workspace.'"
        ]
    elif score < 80:
        if cups_bottles >= 3:
            return "☕ Hydration station detected—doubles as a recycling center!"
        elif food_items >= 2:
            return "🍕 Emergency snack bunker activated—survival mode engaged!"
        roasts = [
            "🌪️ Hurricane season called; wants tips on your technique.",
            "🔍 I spy with my little eye... everything. Everywhere."
        ]
    else:
        roasts = [
            "🔬 Anthropologists would love to study this habitat.",
            "🌋 This chaos level has its own gravitational pull.",
            "📡 NASA called—they can see your mess from space."
        ]
    
    import random
    return random.choice(roasts)

def create_progress_bar(score):
    if score <= 25: color = "#4CAF50"
    elif score <= 50: color = "#FF9800"  
    elif score <= 75: color = "#F44336"
    else: color = "#9C27B0"
    
    return f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0;">
        <div style="width: {score:.1f}%; background-color: {color}; height: 35px; 
                    display: flex; align-items: center; justify-content: center; 
                    color: white; font-weight: bold; transition: width 0.5s ease-in-out; font-size: 16px;">
            {score:.1f}%
        </div>
    </div>
    """

def analyze_image(image, mirror_cam, coffee_mode, model):
    """Main analysis function with comprehensive error handling"""
    try:
        st.write("🔄 Starting analysis...")
        
        # Preprocess image
        bgr, gray = preprocess_image(image, mirror=mirror_cam)
        if bgr is None:
            return None
        
        st.write("✅ Image preprocessed")
        
        # Compute chaos features
        features = compute_chaos_features(gray)
        if features is None:
            return None
        
        st.write("✅ Features computed")
        
        # Detect objects
        detections, obj_feat = detect_objects(bgr, model, 0.35)
        st.write(f"✅ Objects detected: {len(detections)}")
        
        # Compute chaos score
        score, component_scores = compute_chaos_score(features, detections, coffee_mode)
        st.write(f"✅ Chaos score: {score:.1f}")
        
        return {
            "score": score,
            "label": get_chaos_label(score),
            "roast": get_roast(score, detections),
            "component_scores": component_scores,
            "detections": detections,
            "obj_features": obj_feat,
            "annotated_image": draw_detections(bgr, detections),
            "raw_image": bgr,
            "edges": features["edges"],
            "chaos_map": features["chaos_map"]
        }
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.write(f"Debug info: {type(e).__name__}: {e}")
        return None

# Load model
st.write("Loading YOLO model...")
model = load_model()

# UI Layout
st.title("🌪️ Chaos Analyzer ")
st.caption("Chaos analysis to meet the chaos around you!")

# SIDEBAR CONTROLS
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Main Settings
    st.subheader("📷 Input Settings")
    input_mode = st.radio("Input Mode", ["📷 Camera", "📁 Upload"], index=0)
    
    if input_mode == "📷 Camera":
        mirror_live_feed = st.toggle("🪞 Mirror live camera feed", value=True)
        if mirror_live_feed:
            st.success("🪞 Live feed will be mirrored")
        else:
            st.info("📷 Live feed normal view")
    else:
        mirror_live_feed = False
    
    # Judging Mode
    st.subheader("☕ Judging Mode")
    coffee_mode = st.selectbox("Mood", 
        options=[0.8, 0.9, 1.0, 1.1, 1.2],
        index=2,
        format_func=lambda x: {0.8: "Very Gentle", 0.9: "After Coffee", 
                              1.0: "Normal", 1.1: "Before Coffee", 
                              1.2: "Very Critical"}[x]
    )

# CSS mirroring
if input_mode == "📷 Camera" and mirror_live_feed:
    st.markdown("""
    <style>
    [data-testid="stCameraInput"] video {
        transform: scaleX(-1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# MAIN INTERFACE
camera_col, results_col = st.columns([1, 1])

# LEFT PANEL - CAMERA VIEW
with camera_col:
    st.markdown("## 📸 Camera View")
    
    if input_mode == "📷 Camera":
        st.info("📷 **Debug Mode**: Enhanced error reporting enabled!")
        
        camera_key = f"camera_debug_{st.session_state.photo_counter}"
        camera_photo = st.camera_input("📸 Take Picture", key=camera_key)
        
        if camera_photo is not None:
            current_image = Image.open(camera_photo)
            
            display_image = mirror_image_if_needed(current_image, mirror_live_feed)
            caption_text = "📸 Captured Photo (Mirrored)" if mirror_live_feed else "📸 Captured Photo"
            st.image(display_image, caption=caption_text, use_container_width=True)
            
            photo_time = time.time()
            if photo_time - st.session_state.last_analysis_time > 1.0:
                with st.spinner("🔬 Analyzing with debug info..."):
                    result = analyze_image(current_image, mirror_live_feed, coffee_mode, model)
                    if result:
                        st.session_state.analysis_result = result
                        st.session_state.last_analysis_time = photo_time
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("❌ Analysis failed - check error messages above")
                
                st.session_state.photo_counter += 1
                time.sleep(0.3)
                st.rerun()
    
    else:  # Upload mode
        st.info("📁 **Upload Mode**: Select image file")
        
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                current_image = Image.open(uploaded_file)
                st.image(current_image, caption="📁 Uploaded Image", use_container_width=True)
                
                with st.spinner("🔬 Analyzing uploaded image..."):
                    result = analyze_image(current_image, False, coffee_mode, model)
                    if result:
                        st.session_state.analysis_result = result
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("❌ Analysis failed - check error messages above")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

# RIGHT PANEL - RESULTS VIEW  
with results_col:
    st.markdown("## 📊 Analysis Results")
    
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Main score display
        st.markdown(f"### {result['label']}")
        st.markdown(create_progress_bar(result["score"]), unsafe_allow_html=True)
        
        # Status indicator
        if result["score"] <= 25:
            st.success("🎯 Excellent organization!")
        elif result["score"] <= 50:
            st.info("👍 Moderate chaos")
        elif result["score"] <= 75:
            st.warning("⚠️ High chaos detected")
        else:
            st.error("🚨 Extreme chaos alert!")
        
        # Quick stats
        obj_count = result['obj_features']['object_count']
        st.metric("Objects Detected", obj_count, delta=f"+{obj_count}" if obj_count > 0 else "None")
        
        if obj_count > 0:
            detected_items = list(set([name for _, _, _, _, _, name in result['detections']]))
            st.write(f"**Found:** {', '.join(detected_items[:3])}")
            if len(detected_items) > 3:
                st.caption(f"...and {len(detected_items) - 3} more")
        
        # Roast section
        if st.button("🎭 Get Roasted!", type="primary", use_container_width=True):
            st.markdown(f"### 💬 *{result['roast']}*")
            if result["score"] > 75:
                st.snow()
        
        # Feature breakdown
        st.markdown("#### 📈 Chaos Components")
        component_scores = result["component_scores"]
        
        for name, value in component_scores.items():
            st.write(f"{name.replace('_', ' ').title()}: {value:.1f}")
            st.progress(float(value) / 100.0)
        
        # Visual analysis tabs
        st.markdown("#### 🔍 Visual Analysis")
        
        col_tabs, col_toggle = st.columns([3, 1])
        
        with col_toggle:
            mirror_analysis_output = st.toggle("🔄 Mirror", value=True, 
                                                help="Mirror analysis output", 
                                                key="output_mirror_toggle")
        
        with col_tabs:
            tab1, tab2, tab3 = st.tabs(["🎯 Annotated", "⚡ Edges", "🌡️ Heatmap"])
        
        with tab1:
            annotated_bgr = result["annotated_image"]
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            
            if mirror_analysis_output and input_mode == "📷 Camera":
                annotated_rgb = mirror_image_if_needed(annotated_rgb, True)
            
            st.image(annotated_rgb, caption="🎯 Detected Objects", use_container_width=True)
            
        with tab2:
            edges = result["edges"]
            if mirror_analysis_output and input_mode == "📷 Camera":
                edges = mirror_image_if_needed(edges, True)
            
            st.image(edges, caption="⚡ Edge Map", clamp=True, use_container_width=True)
            
        with tab3:
            chaos_map = result["chaos_map"]
            if chaos_map.size > 0 and chaos_map.max() > chaos_map.min():
                normalized = (chaos_map - chaos_map.min()) / (chaos_map.max() - chaos_map.min())
                colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                if mirror_analysis_output and input_mode == "📷 Camera":
                    colored = mirror_image_if_needed(colored, True)
                
                st.image(colored, caption="🌡️ Chaos Heatmap", use_container_width=True)
            else:
                st.info("🌡️ No variation for heatmap")
        
    else:
        st.info("👆 Take a photo to see results!")
        st.markdown("### ⚫ Feel The Chaos")
        st.markdown("""
        - Chaos Analyzer 
        - Takes Your Image And Analyzes It For Chaos
        - See The Chaos Within You
        """)

st.markdown("---")
st.caption("🎯 Built with ❤️ for hackathons • **Fixed with comprehensive error handling!** 🔧✅")
