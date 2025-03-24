import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pyvis.network import Network
import tempfile
import os
from conceptnet.querries_engine import get_graph_data



# Load YOLO model
model = YOLO('./yolo/yolo11s.pt')

st.set_page_config(page_title="YOLO + Scene Graph", layout="wide")
st.title("🔎 Object Detection with Scene Graph")


def make_graph(used_for_result, is_a_result, at_location_result):
    result = []
    for k in used_for_result:
        result.append((k, "used_for", used_for_result[k]))
    for k in is_a_result:
        result.append((k, "is_a", is_a_result[k][0]))
    for k in at_location_result:
        result.append((k, "at_location", at_location_result[k]))
    return result

def draw_triplet_graph_pyvis(triplets):
    net = Network(height='600px', width='100%', directed=True)
    for src, rel, tgt in triplets:
        net.add_node(src, label=src)
        net.add_node(tgt, label=tgt)
        net.add_edge(src, tgt, label=rel)

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_path.name)
    return tmp_path.name

# --- Object Detection ---
def detect_objects(image):
    results = model(image)
    boxes = results[0].boxes
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detected_labels = []

    for box in boxes:
        coords = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, coords)
        conf = box.conf[0].item() if box.conf is not None else 0.0
        cls = int(box.cls[0].item()) if box.cls is not None else -1
        label = results[0].names[cls] if cls in results[0].names else str(cls)

        detected_labels.append(label)

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, detected_labels

# --- Streamlit UI ---
uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
process_button = st.button("🛠️ Detect Objects and Generate Graph")

if uploaded_image and process_button:
    img = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img)

    # Detect
    image_out, labels = detect_objects(img_np)
    unique_labels = sorted(set([l.replace(" ","_") for l in labels]))
    
    print(unique_labels)

    # Get triplets
    used_for_result, is_a_result, at_location_result = get_graph_data(unique_labels)
    triplets = make_graph(used_for_result, is_a_result, at_location_result)
    graph_html_path = draw_triplet_graph_pyvis(triplets)

    # --- Image + Object List Side-by-Side ---
    col1, col2 = st.columns([3, 1])  # Wider image column

    with col1:
        st.image(image_out, caption="Detected Objects", use_container_width=True)

    with col2:
        st.subheader("🧾 Detected Objects:")
        if unique_labels:
            st.markdown("- " + "\n- ".join(unique_labels))
        else:
            st.write("No objects detected.")

    # --- Scene Graph Below ---
    st.subheader("📊 Scene Graph")
    with open(graph_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

    os.unlink(graph_html_path)
