"""
Chaos Analyzer Pro — Live, object-aware chaos scoring
Dependencies:
  pip install streamlit streamlit-webrtc ultralytics opencv-python scikit-image numpy pillow av

Run:
  streamlit run app.py
"""

import time
import math
import threading
from collections import deque, defaultdict

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.measure import shannon_entropy

# Realtime webcam
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# Object detection
from ultralytics import YOLO

st.set_page_config(page_title="Chaos Analyzer Pro", page_icon="🌪️", layout="wide")

# ---------------------------
# Caching and utilities
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    # Small, CPU-friendly model; will download on first run
    return YOLO("yolov8n.pt")

def preprocess_image_bgr(bgr, target_width=640, mirror=False):
    h, w = bgr.shape[:2]
    new_w = target_width
    new_h = int(h * (target_width / w))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if mirror:
        resized = cv2.flip(resized, 1)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return resized, gray, blurred

def compute_low_level_features(gray, canny_low, canny_high, window=64, stride=32):
    edges = cv2.Canny(gray, canny_low, canny_high)
    ed = float(np.count_nonzero(edges)) / float(edges.size)

    H = float(shannon_entropy(gray))

    h, w = edges.shape
    window_eds = []
    # Precompute chaos map grid shape
    rows = max(1, (h - window) // stride + 1)
    cols = max(1, (w - window) // stride + 1)
    chaos_map = np.zeros((rows, cols), dtype=np.float32)
    i = 0
    for y in range(0, max(1, h - window + 1), stride):
        j = 0
        for x in range(0, max(1, w - window + 1), stride):
            patch = edges[y:y+window, x:x+window]
            if patch.size == 0:
                w_ed = 0.0
            else:
                w_ed = float(np.count_nonzero(patch)) / float(patch.size)
            window_eds.append(w_ed)
            if i < rows and j < cols:
                chaos_map[i, j] = w_ed
            j += 1
        i += 1

    LC = float(np.std(window_eds)) if window_eds else 0.0
    return {
        "edges": edges,
        "edge_density": ed,
        "entropy": H,
        "local_chaos": LC,
        "local_chaos_map": chaos_map
    }

# Messiness weights for classes (fallback default=0.5)
CLASS_WEIGHTS = {
    "cup": 1.0, "bottle": 1.0, "wine glass": 1.0, "bowl": 0.8, "banana": 0.7, "apple": 0.7,
    "orange": 0.7, "broccoli": 0.5, "carrot": 0.6, "pizza": 0.8, "donut": 0.7, "cake": 0.7,
    "chair": 0.6, "couch": 0.6, "potted plant": 0.6, "bed": 0.5, "dining table": 0.6,
    "toilet": 0.5, "tv": 0.4, "laptop": 0.4, "mouse": 0.6, "remote": 0.6, "keyboard": 0.6,
    "cell phone": 0.6, "book": 1.0, "clock": 0.4, "vase": 0.7, "scissors": 0.6, "teddy bear": 0.7,
    "hair drier": 0.7, "toothbrush": 0.7, "backpack": 0.9, "umbrella": 0.8, "handbag": 0.9,
    "tie": 0.8, "suitcase": 0.9, "frisbee": 0.7, "skis": 0.7, "snowboard": 0.7, "sports ball": 0.7,
    "kite": 0.7, "baseball bat": 0.7, "baseball glove": 0.7, "skateboard": 0.8, "surfboard": 0.7,
    "tennis racket": 0.7, "bottle": 1.0, "fork": 0.7, "knife": 0.7, "spoon": 0.7, "bowl": 0.8,
    "sandwich": 0.8, "hot dog": 0.8, "toaster": 0.6, "refrigerator": 0.4, "book": 1.0,
    "clock": 0.4, "vase": 0.7, "scissors": 0.6, "teddy bear": 0.7
}

def compute_object_features(bgr_img, model, conf_thres=0.35, draw=False):
    h, w = bgr_img.shape[:2]
    area = float(h * w)
    results = model.predict(bgr_img, imgsz=640, conf=conf_thres, verbose=False)
    detections = []
    if len(results) > 0:
        r = results[0]
        names = r.names
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(np.float32)
                x1, y1, x2, y2 = xyxy.tolist()
                cls_id = int(b.cls.cpu().item())
                conf = float(b.conf.cpu().item())
                cls_name = names.get(cls_id, str(cls_id))
                detections.append((x1, y1, x2, y2, conf, cls_name))

    # Features
    count = len(detections)
    obj_density = count / max(1.0, area)  # normalized by image area

    # Weighted messiness
    weighted_sum = 0.0
    for x1, y1, x2, y2, conf, cname in detections:
        wt = CLASS_WEIGHTS.get(cname, 0.5)
        weighted_sum += wt
    weighted_mess = weighted_sum / max(1.0, area / (640 * 480))  # light normalization

    # Dispersion: centroid variance normalized by image diagonal
    if count >= 2:
        centroids = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0] for (x1, y1, x2, y2, _, _) in detections], dtype=np.float32)
        mean = centroids.mean(axis=0, keepdims=True)
        var = ((centroids - mean) ** 2).mean()
        diag = math.hypot(w, h)
        dispersion = float(var) / (diag * diag + 1e-6)
    else:
        dispersion = 0.0

    return detections, {
        "object_count": count,
        "object_density": obj_density,
        "weighted_mess": weighted_mess,
        "dispersion": dispersion
    }

def normalize_feature(x, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100.0)

def compute_score(low, obj, bounds, weights, coffee=1.0):
    ed_norm = normalize_feature(low["edge_density"], bounds["ed_min"], bounds["ed_max"])
    H_norm = normalize_feature(low["entropy"], bounds["h_min"], bounds["h_max"])
    LC_norm = normalize_feature(low["local_chaos"], bounds["lc_min"], bounds["lc_max"])
    OD_norm = normalize_feature(obj["object_density"], 0.0, bounds["od_max"])
    WM_norm = normalize_feature(obj["weighted_mess"], 0.0, bounds["wm_max"])
    DP_norm = normalize_feature(obj["dispersion"], 0.0, bounds["dp_max"])

    score = (
        weights["w_ed"] * ed_norm +
        weights["w_H"] * H_norm +
        weights["w_LC"] * LC_norm +
        weights["w_OD"] * OD_norm +
        weights["w_WM"] * WM_norm +
        weights["w_DP"] * DP_norm
    ) * coffee

    score = float(np.clip(score, 0, 100))
    return score, {
        "ed_norm": ed_norm, "H_norm": H_norm, "LC_norm": LC_norm,
        "OD_norm": OD_norm, "WM_norm": WM_norm, "DP_norm": DP_norm
    }

def label_for_score(s):
    if s <= 15: return "🧘 Monk Mode"
    if s <= 35: return "✨ Neat Nook"
    if s <= 55: return "🎯 Controlled Chaos"
    if s <= 75: return "🌪️ Hurricane Hover"
    return "👹 Goblin Lair"

def roast_for(score, detections, obj_feats):
    cups = sum(1 for _,_,_,_,_,c in detections if "cup" in c or "bottle" in c)
    books = sum(1 for _,_,_,_,_,c in detections if "book" in c)
    clothes_like = sum(1 for _,_,_,_,_,c in detections if c in {"backpack","handbag","tie","suitcase","teddy bear","pillow"} )

    if score < 20:
        return "Minimalism called; it’s proud of this sanctuary."
    if score < 50:
        if books >= 5:
            return "A library in the wild—Dewey Decimal could help."
        return "Somewhere between zen and ‘I’ll deal with it tomorrow.’"
    if score < 80:
        if cups >= 5:
            return "Hydration nation: this doubles as a recycling center."
        if clothes_like >= 3:
            return "Laundry day is staging a coup."
        return "Your floor is Schrödinger’s desk."
    return "Anthropologists would like to study this habitat."

def color_for_score(s):
    if s <= 25: return "#00c853"
    if s <= 50: return "#ffd600"
    if s <= 75: return "#ff9100"
    return "#ff1744"

def meter_html(score):
    color = color_for_score(score)
    return f"""
    <div style="background:#eee;border-radius:10px;padding:4px;">
      <div style="background:{color};width:{score:.1f}%;height:30px;border-radius:7px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;">
        {score:.1f}%
      </div>
    </div>
    """

def draw_boxes(bgr, detections):
    out = bgr.copy()
    for x1,y1,x2,y2,conf,cls in detections:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 180, 255), 2)
        cv2.putText(out, f"{cls} {conf:.2f}", (p1[0], max(0, p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 3, cv2.LINE_AA)
        cv2.putText(out, f"{cls} {conf:.2f}", (p1[0], max(0, p1[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def heatmap_from_map(ch_map):
    cmap = (ch_map - ch_map.min()) / (max(1e-8, ch_map.max() - ch_map.min()))
    cmap = (cmap * 255).astype(np.uint8)
    cm = cv2.applyColorMap(cmap, cv2.COLORMAP_JET)
    return cm

# ---------------------------
# Sidebar controls
# ---------------------------

st.title("🌪️ Chaos Analyzer Pro")
st.caption("Live, object-aware chaos scoring with explainable features and playful roasts.")

with st.sidebar:
    st.header("🎛️ Controls")

    input_mode = st.radio("Input", ["Upload Image", "Live Webcam"], index=1)

    st.subheader("Low-level features")
    canny_low = st.slider("Canny low", 10, 150, 50, 1)
    canny_high = st.slider("Canny high", 100, 300, 150, 1)
    window_size = st.slider("Local chaos window", 32, 128, 64, 16)
    stride = st.slider("Local chaos stride", 16, 64, 32, 8)

    st.subheader("Object detection")
    conf_thres = st.slider("YOLO conf threshold", 0.1, 0.8, 0.35, 0.05)

    st.subheader("Scoring weights")
    w_ed = st.slider("w: Edge Density", 0.0, 1.0, 0.35, 0.05)
    w_H = st.slider("w: Entropy", 0.0, 1.0, 0.20, 0.05)
    w_LC = st.slider("w: Local Chaos", 0.0, 1.0, 0.10, 0.05)
    w_OD = st.slider("w: Object Density", 0.0, 1.0, 0.20, 0.05)
    w_WM = st.slider("w: Weighted Messiness", 0.0, 1.0, 0.10, 0.05)
    w_DP = st.slider("w: Dispersion", 0.0, 1.0, 0.05, 0.05)

    st.subheader("Normalization ceilings (advanced)")
    od_max = st.number_input("ObjDensity ceiling", value=0.002, format="%.6f")
    wm_max = st.number_input("WeightedMess ceiling", value=0.004, format="%.6f")
    dp_max = st.number_input("Dispersion ceiling", value=0.060, format="%.3f")

    st.subheader("Judging mode")
    coffee_mode = st.selectbox("Mood", options=[1.1, 1.0, 0.9], index=1, format_func=lambda x: "Before Coffee (Harsh)" if x==1.1 else ("Normal" if x==1.0 else "After Coffee (Gentle)"))

    st.subheader("Calibration")
    st.caption("Capture a few frames to auto-tune normalization of low-level features.")
    calib_clean = st.button("Capture Clean sample")
    calib_messy = st.button("Capture Messy sample")
    calib_reset = st.button("Reset calibration")

show_edges = st.checkbox("Show edge map", value=False)
show_heatmap = st.checkbox("Show local chaos heatmap", value=False)
show_boxes = st.checkbox("Show object boxes", value=True)

mirror_webcam = st.checkbox("Mirror webcam view", value=True) if input_mode == "Live Webcam" else False
auto_analyze = st.checkbox("Auto analyze (continuous)", value=True) if input_mode == "Live Webcam" else False
analyze_now = st.button("Analyze current frame") if input_mode == "Live Webcam" and not auto_analyze else False

# Default normalization bounds for low-level features
bounds = {
    "ed_min": 0.02, "ed_max": 0.15,
    "h_min": 4.00, "h_max": 7.50,
    "lc_min": 0.01, "lc_max": 0.06,
    "od_max": float(od_max), "wm_max": float(wm_max), "dp_max": float(dp_max)
}

weights = {"w_ed": w_ed, "w_H": w_H, "w_LC": w_LC, "w_OD": w_OD, "w_WM": w_WM, "w_DP": w_DP}

# Calibration buffers
if "calib_clean" not in st.session_state:
    st.session_state.calib_clean = {"ed": [], "H": [], "LC": []}
if "calib_messy" not in st.session_state:
    st.session_state.calib_messy = {"ed": [], "H": [], "LC": []}

def apply_calibration(low):
    # Update bounds based on collected samples
    clean = st.session_state.calib_clean
    messy = st.session_state.calib_messy
    ed_vals = clean["ed"] + messy["ed"]
    H_vals = clean["H"] + messy["H"]
    LC_vals = clean["LC"] + messy["LC"]
    if len(ed_vals) >= 2:
        bounds["ed_min"] = float(np.percentile(ed_vals, 5))
        bounds["ed_max"] = float(np.percentile(ed_vals, 95))
    if len(H_vals) >= 2:
        bounds["h_min"] = float(np.percentile(H_vals, 5))
        bounds["h_max"] = float(np.percentile(H_vals, 95))
    if len(LC_vals) >= 2:
        bounds["lc_min"] = float(np.percentile(LC_vals, 5))
        bounds["lc_max"] = float(np.percentile(LC_vals, 95))

# ---------------------------
# Upload mode
# ---------------------------

def analyze_and_display_bgr(bgr, model):
    # Preprocess
    resized, gray, blurred = preprocess_image_bgr(bgr, 640, mirror=(mirror_webcam and input_mode=="Live Webcam"))
    # Low-level features
    low = compute_low_level_features(gray, canny_low, canny_high, window=window_size, stride=stride)
    # Object features
    dets, obj = compute_object_features(resized, model, conf_thres=conf_thres)

    # Apply calibration if any samples collected
    apply_calibration(low)

    # Score
    score, norms = compute_score(low, obj, bounds, weights, coffee=float(coffee_mode))

    # Visuals
    disp_rows = st.columns([1,1])
    with disp_rows[0]:
        st.markdown(f"### {label_for_score(score)}")
        st.markdown(meter_html(score), unsafe_allow_html=True)
        st.write(f"Edge density: {norms['ed_norm']:.1f} | Entropy: {norms['H_norm']:.1f} | Local chaos: {norms['LC_norm']:.1f}")
        st.write(f"Obj density: {norms['OD_norm']:.1f} | Weighted mess: {norms['WM_norm']:.1f} | Dispersion: {norms['DP_norm']:.1f}")
        st.write(f"Detected objects: {obj['object_count']}")
        st.success(roast_for(score, dets, obj))

    vis = resized.copy()
    if show_boxes and len(dets) > 0:
        vis = draw_boxes(vis, dets)

    with disp_rows[1]:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Annotated view", use_container_width=True)

    # Aux overlays
    aux_cols = st.columns([1,1])
    if show_edges:
        with aux_cols[0]:
            st.image(low["edges"], caption="Edge map", clamp=True, use_container_width=True)
    if show_heatmap and low["local_chaos_map"].size > 0:
        with aux_cols[1]:
            hm = heatmap_from_map(low["local_chaos_map"])
            hm = cv2.resize(hm, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST)
            st.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), caption="Local chaos heatmap", use_container_width=True)

    # Calibration capture
    if calib_clean:
        st.session_state.calib_clean["ed"].append(low["edge_density"])
        st.session_state.calib_clean["H"].append(low["entropy"])
        st.session_state.calib_clean["LC"].append(low["local_chaos"])
        st.toast("Captured CLEAN sample", icon="✅")
    if calib_messy:
        st.session_state.calib_messy["ed"].append(low["edge_density"])
        st.session_state.calib_messy["H"].append(low["entropy"])
        st.session_state.calib_messy["LC"].append(low["local_chaos"])
        st.toast("Captured MESSY sample", icon="🧹")
    if calib_reset:
        st.session_state.calib_clean = {"ed": [], "H": [], "LC": []}
        st.session_state.calib_messy = {"ed": [], "H": [], "LC": []}
        st.toast("Calibration reset", icon="♻️")

if input_mode == "Upload Image":
    file = st.file_uploader("Upload a room image", type=["jpg","jpeg","png"])
    if file is not None:
        pil = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        model = load_yolo_model()
        analyze_and_display_bgr(bgr, model)

# ---------------------------
# Live webcam mode via WebRTC
# ---------------------------

class ChaosProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_yolo_model()
        self.last_vis = None
        self.last_score = 0.0
        self.last_label = ""
        self.last_roast = ""
        self.last_low = None
        self.last_norms = None
        self.last_dets = []
        self.last_obj = {}
        self.frame_interval = 1/12.0  # target ~12 FPS
        self._last_time = 0.0

    def recv(self, frame):
        # Throttle FPS
        t = time.time()
        if t - self._last_time < self.frame_interval:
            return frame
        self._last_time = t

        img = frame.to_ndarray(format="bgr24")
        # Analyze
        resized, gray, blurred = preprocess_image_bgr(img, 640, mirror=mirror_webcam)
        low = compute_low_level_features(gray, canny_low, canny_high, window=window_size, stride=stride)

        dets, obj = compute_object_features(resized, self.model, conf_thres=conf_thres)
        apply_calibration(low)

        score, norms = compute_score(low, obj, bounds, weights, coffee=float(coffee_mode))

        vis = resized.copy()
        if show_boxes and len(dets) > 0:
            vis = draw_boxes(vis, dets)

        self.last_vis = vis
        self.last_score = score
        self.last_label = label_for_score(score)
        self.last_roast = roast_for(score, dets, obj)
        self.last_low = low
        self.last_norms = norms
        self.last_dets = dets
        self.last_obj = obj

        # If auto analyze disabled, still return preview but computations already done here
        return av.VideoFrame.from_ndarray(vis, format="bgr24")

# WebRTC config (public STUN server)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if input_mode == "Live Webcam":
    st.info("Grant camera permission to start live analysis. Use Auto analyze for continuous scoring, or click Analyze current frame.")
    ctx = webrtc_streamer(
        key="chaos-analyzer-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=ChaosProcessor,
        async_processing=True,
    )

    if ctx.video_processor:
        vp: ChaosProcessor = ctx.video_processor

        # In auto mode, frames are analyzed continuously in recv()
        # If not auto, we still process frames but update UI on button press
        placeholder = st.empty()

        def render_from_state():
            if vp.last_vis is None:
                return
            cols = st.columns([1,1])
            with cols[0]:
                st.markdown(f"### {vp.last_label}")
                st.markdown(meter_html(vp.last_score), unsafe_allow_html=True)
                if vp.last_norms:
                    st.write(f"Edge density: {vp.last_norms['ed_norm']:.1f} | Entropy: {vp.last_norms['H_norm']:.1f} | Local chaos: {vp.last_norms['LC_norm']:.1f}")
                    st.write(f"Obj density: {vp.last_norms['OD_norm']:.1f} | Weighted mess: {vp.last_norms['WM_norm']:.1f} | Dispersion: {vp.last_norms['DP_norm']:.1f}")
                if vp.last_obj:
                    st.write(f"Detected objects: {vp.last_obj.get('object_count',0)}")
                st.success(vp.last_roast or "")
            with cols[1]:
                st.image(cv2.cvtColor(vp.last_vis, cv2.COLOR_BGR2RGB), caption="Live annotated view", use_container_width=True)

            aux_cols = st.columns([1,1])
            if show_edges and vp.last_low is not None:
                with aux_cols[0]:
                    st.image(vp.last_low["edges"], caption="Edge map", clamp=True, use_container_width=True)
            if show_heatmap and vp.last_low is not None and vp.last_low["local_chaos_map"].size > 0:
                with aux_cols[1]:
                    hm = heatmap_from_map(vp.last_low["local_chaos_map"])
                    hm = cv2.resize(hm, (vp.last_vis.shape[1], vp.last_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                    st.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), caption="Local chaos heatmap", use_container_width=True)

        if auto_analyze:
            # Periodically refresh UI
            # Use a lightweight loop triggered by reruns; Streamlit runs top-to-bottom,
            # so we just display the latest state each pass.
            render_from_state()
        else:
            if analyze_now:
                render_from_state()

        # Calibration capture
        if calib_clean and vp.last_low is not None:
            st.session_state.calib_clean["ed"].append(vp.last_low["edge_density"])
            st.session_state.calib_clean["H"].append(vp.last_low["entropy"])
            st.session_state.calib_clean["LC"].append(vp.last_low["local_chaos"])
            st.toast("Captured CLEAN sample", icon="✅")
        if calib_messy and vp.last_low is not None:
            st.session_state.calib_messy["ed"].append(vp.last_low["edge_density"])
            st.session_state.calib_messy["H"].append(vp.last_low["entropy"])
            st.session_state.calib_messy["LC"].append(vp.last_low["local_chaos"])
            st.toast("Captured MESSY sample", icon="🧹")
        if calib_reset:
            st.session_state.calib_clean = {"ed": [], "H": [], "LC": []}
            st.session_state.calib_messy = {"ed": [], "H": [], "LC": []}
            st.toast("Calibration reset", icon="♻️")

st.markdown("---")
st.caption("Tip: Adjust weights and ceilings to make the score feel less arbitrary in your environment. Use calibration to set realistic ranges.")
