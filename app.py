import os
import json
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models.classifier import load_checkpoint
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.predict import get_inference_transform

st.set_page_config(
    page_title="Flower Image Classifier | Transfer Learning",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 10px;
    }
    .pred-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "models/best_model.pth"
SAMPLE_DIR = "sample_images"
REPORTS_DIR = "reports"

@st.cache_resource
def get_cached_model():
    """Loads and caches the PyTorch model checkpoint."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(MODEL_PATH, device=device)
    return model, checkpoint

def main():
    st.markdown('<div class="main-header">🌸 Flower Image Classifier with Transfer Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fine-tuned Deep Convolutional Neural Network (MobileNetV2 / PyTorch) for 5 Flower Classes</div>', unsafe_allow_html=True)

    # Sidebar: Model and System Info
    st.sidebar.title("⚙️ Model Information")
    model, checkpoint = get_cached_model()

    if model is None:
        st.sidebar.warning("⚠️ Model weights not found at `models/best_model.pth`. Please run training first.")
    else:
        model_name = checkpoint.get("model_name", "mobilenet_v2").upper()
        classes = checkpoint.get("class_names", [])
        best_val = checkpoint.get("val_accuracy", 0.0)
        
        st.sidebar.success(f"✅ **Loaded Model**: `{model_name}`")
        st.sidebar.metric("Target Classes", len(classes))
        st.sidebar.metric("Validation Accuracy", f"{best_val * 100:.2f}%")
        st.sidebar.write("**Classes:**", ", ".join([c.capitalize() for c in classes]))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Dataset Details")
    st.sidebar.info("""
    - **Dataset**: Google/TensorFlow Flowers
    - **Splits**: 70% Train, 15% Val, 15% Test
    - **Augmentation**: Random Crop, Rotation, Horizontal Flip, Color Jitter
    - **Pretrained Weights**: ImageNet
    """)

    # Main Tabs
    tab_predict, tab_analytics, tab_docs = st.tabs(["🎯 Image Prediction", "📊 Model Performance & Metrics", "📖 Methodology & Limitations"])

    # TAB 1: PREDICTION
    with tab_predict:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("1. Select or Upload Image")
            
            input_mode = st.radio("Choose Input Mode:", ["Upload Image File", "Use Sample Test Images"], horizontal=True)
            image_to_process = None

            if input_mode == "Upload Image File":
                uploaded_file = st.file_uploader("Upload a flower photo (JPG, PNG, JPEG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
                if uploaded_file is not None:
                    try:
                        image_to_process = Image.open(uploaded_file).convert("RGB")
                    except Exception as e:
                        st.error(f"Error opening image: {e}")
            else:
                sample_files = {
                    "Daisy 🌼": "daisy.jpg",
                    "Dandelion 🌾": "dandelion.jpg",
                    "Rose 🌹": "roses.jpg",
                    "Sunflower 🌻": "sunflowers.jpg",
                    "Tulip 🌷": "tulips.jpg"
                }
                selected_sample = st.selectbox("Select a sample flower:", list(sample_files.keys()))
                sample_path = os.path.join(SAMPLE_DIR, sample_files[selected_sample])
                if os.path.exists(sample_path):
                    image_to_process = Image.open(sample_path).convert("RGB")
                else:
                    st.warning(f"Sample file not found at {sample_path}")

            if image_to_process is not None:
                st.image(image_to_process, caption="Input Image Preview", use_container_width=True)

        with col_output:
            st.subheader("2. Classification Results")
            if image_to_process is not None:
                if model is None:
                    st.error("Model is not loaded. Train the model first to perform inference.")
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    classes = checkpoint.get("class_names", [])
                    
                    # Preprocess and infer
                    transform = get_inference_transform()
                    tensor = transform(image_to_process).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                    top_idx = int(probs.argmax())
                    top_class = classes[top_idx].capitalize()
                    top_confidence = probs[top_idx] * 100.0

                    # Pred Card
                    st.markdown(f"""
                    <div class="pred-card">
                        <h4 style="margin:0; color:#1E40AF;">Top Prediction</h4>
                        <h1 style="margin:5px 0; color:#1D4ED8; font-size:2.4rem;">{top_class}</h1>
                        <p style="margin:0; font-size:1.2rem; font-weight:600; color:#047857;">Confidence: {top_confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probabilities Bar Chart
                    df_probs = pd.DataFrame({
                        "Class": [c.capitalize() for c in classes],
                        "Confidence (%)": [round(p * 100.0, 2) for p in probs]
                    }).sort_values(by="Confidence (%)", ascending=True)

                    st.markdown("##### Class Probability Distribution")
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    bars = ax.barh(df_probs["Class"], df_probs["Confidence (%)"], color="#3B82F6", edgecolor="#1D4ED8", height=0.6)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Confidence (%)", fontsize=10)
                    ax.grid(axis='x', linestyle='--', alpha=0.5)

                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                                va='center', ha='left', fontsize=9, fontweight='bold', color="#1F2937")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Data table
                    with st.expander("🔍 View Raw Probability Breakdown"):
                        st.dataframe(df_probs.sort_values(by="Confidence (%)", ascending=False), use_container_width=True)
            else:
                st.info("👈 Please upload an image or pick a sample from the left panel to see predictions.")

    # TAB 2: ANALYTICS & METRICS
    with tab_analytics:
        st.subheader("Model Evaluation & Training Metrics")
        
        # Load results json if available
        results_file = os.path.join(REPORTS_DIR, "evaluation_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Test Accuracy", f"{results['test_accuracy']*100:.2f}%")
            c2.metric("Test Loss", f"{results['test_loss']:.4f}")
            c3.metric("Test Samples Evaluated", results['total_samples'])
            c4.metric("Total Classes", len(results['class_names']))

            st.markdown("---")

        col_cm, col_curves = st.columns(2, gap="medium")
        cm_file = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        curves_file = os.path.join(REPORTS_DIR, "training_curves.png")

        with col_cm:
            st.markdown("#### Normalized Confusion Matrix")
            if os.path.exists(cm_file):
                st.image(cm_file, caption="Confusion Matrix on Test Split", use_container_width=True)
            else:
                st.info("Confusion matrix plot will appear once evaluation completes.")

        with col_curves:
            st.markdown("#### Training & Validation Learning Curves")
            if os.path.exists(curves_file):
                st.image(curves_file, caption="Loss and Accuracy over Epochs", use_container_width=True)
            else:
                st.info("Training curves will appear once training completes.")

        if os.path.exists(results_file):
            st.markdown("#### Per-Class Classification Report")
            cr = results.get("classification_report", {})
            cr_df = pd.DataFrame(cr).transpose()
            st.dataframe(cr_df.style.format(precision=4), use_container_width=True)

    # TAB 3: DOCUMENTATION & LIMITATIONS
    with tab_docs:
        st.subheader("Dataset Documentation, Methodology, & Model Limitations")
        st.markdown("""
        ### 1. Dataset Origin & Composition
        The dataset consists of photographs across 5 floral categories sourced from Google / TensorFlow Example Datasets:
        - **Classes (5)**: `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`
        - **Total Images**: 3,670 natural images in varying outdoor and indoor lighting conditions.
        - **Split Strategy**: Stratified partitioning into:
          - **Train (70%)**: 2,567 images
          - **Validation (15%)**: 547 images
          - **Test (15%)**: 556 images

        ### 2. Data Augmentation
        To mitigate overfitting on small datasets, robust real-time augmentations were applied to the training split:
        - `RandomResizedCrop(224, scale=(0.8, 1.0))`: Invariance to object scale and cropping
        - `RandomHorizontalFlip(p=0.5)`: Mirror invariance
        - `RandomRotation(degrees=15)`: Orientation variance
        - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: Lighting and exposure robustness
        - `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: Standard ImageNet normalization

        ### 3. Transfer Learning Pipeline
        - **Architecture**: MobileNetV2 pretrained on ImageNet-1k (~3.5M parameters, high efficiency).
        - **Phase 1 (Feature Extraction)**: Frozen convolutional backbone with learning rate $10^{-3}$ and AdamW optimizer. Only the custom dropout + 2-layer classification head is trained.
        - **Phase 2 (Fine-Tuning)**: Top inverted residual blocks unfrozen with Cosine Annealing learning rate schedule ($10^{-4}$ down to $10^{-6}$) to adapt high-level convolutional features to domain-specific flower patterns.

        ### 4. Known Limitations & Failure Modes
        1. **Inter-class Morphological Similarity**: Yellow dandelion flower heads can occasionally be confused with yellow daisies or young sunflowers due to color dominance.
        2. **Multi-object Scenes**: If an image contains multiple flower species simultaneously, the model will output a single dominant label.
        3. **Background Clutter & Foliage**: Heavy green leaf backgrounds with small blooms may decrease model confidence.
        4. **Out-of-Distribution (OOD) Classes**: Inputting non-flower images (e.g., cars, animals) will force a prediction among the 5 trained flower classes with pseudo-confidence. An explicit OOD / open-set thresholding filter is recommended for production deployment.
        """)

if __name__ == "__main__":
    main()
