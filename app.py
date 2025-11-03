"""
Aviation-Themed Food Classification Demo with Streamlit
Fighter Jet Inspired UI 🛩️
"""

import base64  # Add this import for image handling
# Install streamlit if needed
try:
    import streamlit as st
except:
    import subprocess
    subprocess.run(["pip", "install", "-q", "streamlit"], check=True)
    import streamlit as st

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel, PeftConfig
import json
import os

print("🛩️ Initializing Aviation Food Classifier...")

# ============================================================================
# LOAD MODEL AND CONFIG
# ============================================================================

OUTPUT_DIR = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open(os.path.join(OUTPUT_DIR, "label2id.json"), 'r') as f:
    label2id = json.load(f)

with open(os.path.join(OUTPUT_DIR, "id2label.json"), 'r') as f:
    id2label_json = json.load(f)
    id2label = {int(k): v for k, v in id2label_json.items()}

# Load processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Load model with LoRA
print("📥 Loading trained model...")
base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(id2label),
    ignore_mismatched_sizes=True
)

peft_config = PeftConfig.from_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
model = PeftModel.from_pretrained(base_model, os.path.join(OUTPUT_DIR, "final_model"), config=peft_config)
model = model.merge_and_unload()
model = model.to(device)
model.eval()

print("✅ Model loaded and ready for deployment!")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def classify_food(image):
    """
    Classify food image and return predictions
    """
    if image is None:
        return None, "⚠️ **ALERT:** No image detected. Please upload an image for classification."
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        
        # Process image
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs[0], 5)
        
        # Create results dictionary for label component
        predictions = {
            id2label[idx.item()].replace('_', ' ').title(): float(prob.item())
            for idx, prob in zip(top5_indices, top5_probs)
        }
        
        # Create detailed text output
        top_class = id2label[top5_indices[0].item()].replace('_', ' ').title()
        top_conf = top5_probs[0].item()
        
        # Create custom HTML for the prediction display
        prediction_html = f"""
        <div style="text-align: center; margin: 2rem 0;">
            <div style="color: #ff6b35; font-size: 2.5em; font-weight: bold; margin-bottom: 1rem;">
                🛩️ MISSION STATUS: TARGET ACQUIRED
            </div>
            <div style="background: linear-gradient(135deg, rgba(30, 60, 114, 0.9), rgba(42, 82, 152, 0.9));
                        border: 4px solid #ff6b35;
                        border-radius: 15px;
                        padding: 2rem;
                        margin: 1rem 0;
                        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);">
                <div style="color: #f7931e; font-size: 1.8em; font-weight: bold; margin-bottom: 1.5rem;">
                    🎯 PRIMARY TARGET IDENTIFIED
                </div>
                <div style="background: linear-gradient(90deg, rgba(255, 107, 53, 0.2), rgba(247, 147, 30, 0.2));
                            padding: 1.5rem;
                            border-radius: 10px;
                            margin: 1rem 0;">
                    <div style="color: white; font-size: 3em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                        {top_class}
                    </div>
                    <div style="color: #f7931e; font-size: 1.5em; margin-top: 1rem;">
                        Confidence Level: {top_conf:.2%} ✈️
                    </div>
                </div>
            </div>
        </div>
        """
        
        # Display the custom HTML
        st.markdown(prediction_html, unsafe_allow_html=True)
        
        result_text = f"""
## 📊 RADAR DETECTION - TOP 5 TARGETS

"""
        
        for i, (class_name, prob) in enumerate(predictions.items(), 1):
            # Create progress bar
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            # Add rank emoji
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            
            result_text += f"{rank_emoji} **{class_name}**\n"
            result_text += f"   `{bar}` {prob:.2%}\n\n"
        
        result_text += f"""
---

## ✈️ FLIGHT STATISTICS
- **Model:** Vision Transformer (ViT) with LoRA
- **Accuracy:** 84.78% on test set
- **Classes:** {len(id2label)} food categories
- **Status:** ✅ MISSION SUCCESSFUL

🛩️ *Powered by Advanced AI Navigation Systems*
"""
        
        return predictions, result_text
        
    except Exception as e:
        error_msg = f"""
# ⚠️ SYSTEM ERROR

**Error Code:** Classification Failure  
**Details:** {str(e)}

Please try again with a different image.
"""
        return None, error_msg

# ============================================================================
# STREAMLIT INTERFACE - AVIATION THEME
# ============================================================================

st.set_page_config(
    page_title="🛩️ Aviation Food Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for the logo and prediction display
st.markdown("""
    <style>
    .logo-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        padding: 10px;
        border-radius: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: -10px;
    }
    .logo-container img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }
    .prediction-frame {
        background: rgba(255, 255, 255, 0.1);
        border: 3px solid #ff6b35;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    .prediction-title {
        font-size: 24px;
        font-weight: bold;
        color: #ff6b35;
        margin-bottom: 15px;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .prediction-content {
        text-align: center;
    }
    .food-name {
        font-size: 36px;
        font-weight: bold;
        color: #ffffff;
        margin: 15px 0;
        padding: 10px;
        background: linear-gradient(90deg, rgba(255, 107, 53, 0.2) 0%, rgba(247, 147, 30, 0.2) 100%);
        border-radius: 8px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .confidence {
        font-size: 20px;
        color: #f7931e;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Create two columns for the header with more space for the logo
header_left, header_right = st.columns([2.5, 1.5])

with header_left:
    st.title("🛩️ AVIATION FOOD CLASSIFICATION SYSTEM")
    st.markdown("""
    ### *Fighter Jet Precision AI - Vision Transformer Technology*
    """)

with header_right:
    # Display the fighter jet logo with styling
    try:
        logo_path = "assets/fighter_jet_logo.png"
        st.markdown(f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}" alt="Fighter Jet Logo">
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.write("🛩️")  # Fallback emoji if image is not found

st.markdown("""
---

**MISSION BRIEFING:** Upload a food image for instant classification using military-grade AI vision systems.  
**CLEARANCE LEVEL:** 84.78% Accuracy | 101 Food Classes | LoRA-Enhanced ViT

---
""")

uploaded_file = st.file_uploader("📡 IMAGE UPLOAD STATION", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    predictions, result_text = classify_food(uploaded_file)
    
    if predictions:
        st.markdown(result_text)
    else:
        st.error(result_text)
else:
    st.info("Please upload an image to begin classification.")

st.sidebar.markdown("""
### 🛩️ SYSTEM SPECIFICATIONS

| Component | Specification |
|-----------|--------------|
| **Architecture** | Vision Transformer (ViT-Base-Patch16-224) |
| **Optimization** | LoRA (Low-Rank Adaptation) |
| **Training** | FP16 Mixed Precision + Gradient Accumulation |
| **Accuracy** | 84.78% on 15,150 test images |
| **Categories** | 101 Food Classes |
| **Deployment** | Real-time GPU Inference |

---

🎖️ **DEVELOPED BY:** [Your Name]  
🛩️ **PROJECT:** Vision Transformer Fine-Tuning  
⚡ **POWERED BY:** Hugging Face Transformers & Streamlit

---

*"Precision in the skies, accuracy in AI"* ✈️
""")