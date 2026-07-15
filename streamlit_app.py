import streamlit as st
import cv2
import numpy as np
import torch
import os
from PIL import Image
from unet import UNet
import gc

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="CrackSeg | Structural Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium glassmorphism aesthetic
st.markdown("""
    <style>
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphism Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(139, 92, 246, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.5);
    }
    
    /* Metric Value Text */
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    
    [data-testid="stMetricLabel"] {
        white-space: nowrap !important;
        font-size: 14px !important;
    }
    
    /* Headers */
    h1 {
        text-align: center !important;
        background: linear-gradient(90deg, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        padding-bottom: 20px;
    }
    
    /* Health Banners */
    .health-banner {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 30px;
        font-weight: 800;
        font-size: 26px;
        letter-spacing: 1px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
    }
    .health-minor { 
        background: rgba(234, 179, 8, 0.15); 
        color: #fde047; 
        border: 1px solid rgba(250, 204, 21, 0.5); 
    }
    .health-moderate { 
        background: rgba(249, 115, 22, 0.15); 
        color: #fdba74; 
        border: 1px solid rgba(251, 146, 60, 0.5); 
    }
    .health-critical { 
        background: rgba(239, 68, 68, 0.15); 
        color: #fca5a5; 
        border: 1px solid rgba(248, 113, 113, 0.5); 
        animation: pulse 2s infinite;
    }
    
    /* Critical Pulse Animation */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.4);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        padding: 10px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(139, 92, 246, 0.2) !important;
        color: #c084fc !important;
        border: 1px solid rgba(139, 92, 246, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Model & Constants
# ------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "best_model.pth"
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # Reassemble model if it was split for GitHub upload
    if not os.path.exists(MODEL_PATH):
        part_num = 0
        if os.path.exists(f"{MODEL_PATH}.part{part_num}"):
            with open(MODEL_PATH, 'wb') as outfile:
                while os.path.exists(f"{MODEL_PATH}.part{part_num}"):
                    with open(f"{MODEL_PATH}.part{part_num}", 'rb') as infile:
                        outfile.write(infile.read())
                    part_num += 1
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    else:
        st.error(f"Model file {MODEL_PATH} not found!")
        return None

model = load_model()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2822/2822765.png", width=60)
    st.title("CrackSeg")
    st.markdown("AI Structural Health Monitor")
    st.divider()
    
    st.markdown("### How it works")
    st.markdown("""
    CrackSeg uses a custom **U-Net** to segment cracks at pixel-level precision.
    
    **Integrity Score Factors:**
    - **Area:** Surface coverage
    - **Count:** Distinct crack regions
    - **Length:** Span of longest crack
    - **Width:** Morphological thickness
    - **Branching:** Intersecting networks
    """)
    st.divider()
    st.markdown("Built with **Streamlit** & **PyTorch**")

# ------------------------------------------------
# Main App Layout
# ------------------------------------------------
st.title("AI Crack Detection")
st.markdown("Upload concrete surface images for instant structural health assessment.")

uploaded_file = st.file_uploader("Choose a concrete image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load Image
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    
    # --- PRE-PROCESS: Scale down for Cloud ---
    max_dim = 512
    h, w = img_np.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_process = cv2.resize(img_np, (new_w, new_h))
    else:
        img_process = img_np.copy()
        
    h_orig, w_orig = img_process.shape[:2]
    
    del image
    del img_np
    gc.collect()

    # 2. Run Inference
    with st.spinner('Running U-Net Inference & Multi-Factor Analysis...'):
        torch.set_num_threads(1)
        
        pad_h = (IMG_SIZE - (h_orig % IMG_SIZE)) % IMG_SIZE
        pad_w = (IMG_SIZE - (w_orig % IMG_SIZE)) % IMG_SIZE
        
        img_padded = cv2.copyMakeBorder(img_process, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        h_pad, w_pad = img_padded.shape[:2]
        
        full_pred = np.zeros((h_pad, w_pad), dtype=np.float32)
        
        for y in range(0, h_pad, IMG_SIZE):
            for x in range(0, w_pad, IMG_SIZE):
                patch = img_padded[y:y+IMG_SIZE, x:x+IMG_SIZE]
                patch_norm = patch.astype(np.float32) / 255.0
                for i in range(3):
                    patch_norm[:, :, i] = (patch_norm[:, :, i] - MEAN[i]) / STD[i]
                
                patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                
                with torch.no_grad():
                    output = model(patch_tensor)
                    pred_patch = torch.sigmoid(output).squeeze().cpu().numpy()
                
                full_pred[y:y+IMG_SIZE, x:x+IMG_SIZE] = pred_patch
                
                del patch_tensor, output
                gc.collect()
                
        pred = full_pred[:h_orig, :w_orig]
        mask = (pred > 0.5)
        mask_uint8_raw = (mask.astype(np.uint8)) * 255
        
        del img_padded, full_pred
        gc.collect()
        
        # 3. Multi-Factor Algorithm
        total_pixels = pred.shape[0] * pred.shape[1]
        crack_pixels = int(mask.sum())
        crack_ratio = (crack_pixels / total_pixels) * 100
        
        area_score = min(crack_ratio / 5.0, 1.0) * 20
        
        num_labels, labels = cv2.connectedComponents(mask_uint8_raw)
        num_cracks = max(0, num_labels - 1)
        crack_count_score = min(num_cracks / 10.0, 1.0) * 15
        
        max_crack_length = 0
        img_diagonal = np.sqrt(w_orig**2 + h_orig**2)
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            coords = cv2.findNonZero(component)
            if coords is not None:
                x_coord, y_coord, w_coord, h_coord = cv2.boundingRect(coords)
                diag = np.sqrt(w_coord**2 + h_coord**2)
                max_crack_length = max(max_crack_length, diag)
        length_score = min(max_crack_length / img_diagonal, 1.0) * 25
        
        avg_width = 0.0
        if crack_pixels > 0:
            skeleton = cv2.ximgproc.thinning(mask_uint8_raw) if hasattr(cv2, 'ximgproc') else mask_uint8_raw
            skeleton_pixels = (skeleton > 0).sum()
            avg_width = crack_pixels / skeleton_pixels if skeleton_pixels > 0 else 1.0
        width_score = min(avg_width / 8.0, 1.0) * 20
        
        branching_score = 0.0
        if num_cracks >= 3:
            large_cracks = sum(1 for label_id in range(1, num_labels) if (labels == label_id).sum() > total_pixels * 0.005)
            branching_score = min(large_cracks / 4.0, 1.0) * 20
            
        damage_score = area_score + crack_count_score + length_score + width_score + branching_score
        integrity_score = max(0, 100 - damage_score)
        
        if crack_pixels == 0:
            status_text = "Healthy"
            status_class = "health-minor"
        elif integrity_score >= 75:
            status_text = "Minor Wear"
            status_class = "health-minor"
        elif integrity_score >= 45:
            status_text = "Moderate Damage"
            status_class = "health-moderate"
        else:
            status_text = "Critical Damage"
            status_class = "health-critical"
            
        # 4. Generate Visuals
        overlay = img_process.copy().astype(float) / 255.0
        overlay[mask, 0] = 1.0
        overlay[mask, 1] = 0.0
        overlay[mask, 2] = 0.0
        
        blended = np.clip(0.6 * (img_process.astype(float)/255.0) + 0.4 * overlay, 0, 1)
        
        heatmap_color = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heatmap_blend = np.clip(0.4 * (img_process.astype(float)/255.0) + 0.6 * (heatmap_rgb.astype(float)/255.0), 0, 1)

    # --- Display Results ---
    st.markdown(f"<div class='health-banner {status_class}'>{status_text} (Score: {integrity_score:.0f}/100)</div>", unsafe_allow_html=True)
    st.progress(int(integrity_score), text="Structural Integrity Score")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Cracks Found", int(num_cracks))
    with col2: st.metric("Max Length", f"{int(max_crack_length)}px")
    with col3: st.metric("Avg Width", f"{avg_width:.1f}px")
    with col4: st.metric("Crack Area", f"{crack_ratio:.2f}%")
    with col5: st.metric("Damage", f"{damage_score:.0f} pts")

    tab1, tab2, tab3, tab4 = st.tabs(["🔴 AI Overlay", "⚫ Binary Mask", "🌡️ Heatmap", "📷 Original"])
    with tab1: st.image(blended, use_container_width=True)
    with tab2: st.image(mask_uint8_raw, use_container_width=True, clamp=True)
    with tab3: st.image(heatmap_blend, use_container_width=True)
    with tab4: st.image(img_process, use_container_width=True)
