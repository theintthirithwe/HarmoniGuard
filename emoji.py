import streamlit as st
import os
import requests
import base64
import io
from PIL import Image
import cv2
import numpy as np
from roboflow import Roboflow


project_url_od = "https://app.roboflow.com/emojidetection-zpoxi/offensive-emoji-detection/13"
private_api_key = "Vftf7wZRWOtU1aXNQDy6"


session_keys = ['include_bbox', 'show_class_label', 'uploaded_file_od', 'amount_blur']
default_values = ["Yes", 'Show Labels', "", "High"]

for key, default in zip(session_keys, default_values):
    if key not in st.session_state:
        st.session_state[key] = default

# Function to run inference
def run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img):
    rf = Roboflow(api_key=private_api_key)
    project = rf.workspace(workspace_id).project(model_id)
    version = project.version(version_number)
    model = version.model



    st.write("#### Uploaded Image")
    st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)


    predictions = model.predict(uploaded_img, overlap=30, confidence=40, stroke=2)
    predictions_json = predictions.json()

    collected_predictions = []
    for bounding_box in predictions:
        x0, y0 = int(bounding_box['x'] - bounding_box['width'] / 2), int(bounding_box['y'] - bounding_box['height'] / 2)
        x1, y1 = int(bounding_box['x'] + bounding_box['width'] / 2), int(bounding_box['y'] + bounding_box['height'] / 2)
        class_name, confidence_score = bounding_box['class'], bounding_box['confidence']
        roi_bbox = [y0, bounding_box['height'], x0, bounding_box['width']]
        collected_predictions.append({"class": class_name, "confidence": confidence_score,
                                      "x0,x1,y0,y1": [x0, x1, y0, y1], "Width": bounding_box['width'],
                                      "Height": bounding_box['height'], "ROI, bbox (y+h,x+w)": roi_bbox,
                                      "bbox area (px)": abs(x0 - x1) * abs(y0 - y1)})

        start_point, end_point = (x0, y0), (x1, y1)
        if st.session_state['include_bbox'] == 'Yes':
            cv2.rectangle(inferenced_img, start_point, end_point, color=(0, 0, 0), thickness=2)  # Set a default thickness

            if st.session_state['show_class_label'] == 'Show Labels':
                # Draw the label outside the bounding box to avoid being blurred
                label_x0 = x0
                label_y0 = y0 - 25 if y0 - 25 > 0 else y1 + 25
                cv2.rectangle(inferenced_img, (label_x0, label_y0), (label_x0 + 100, label_y0 - 20), color=(0, 0, 0), thickness=-1)
                cv2.putText(inferenced_img, f"{class_name} ({confidence_score:.2f})", (label_x0, label_y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    st.write("### Detected Image")
    st.image(inferenced_img, caption="Detected Image", use_container_width=True)

    # Check if any predictions were made and display the appropriate message with animation
    if collected_predictions:
        message = "Offensive Emoji detected"
    else:
        message = "Peaceful!!!"

    # Separate container for the animated text
    with st.container():
        # HTML and CSS for bouncing animation
        html_message = f"""
        <div style="width: 100%; overflow: hidden; white-space: nowrap;">
            <div style="display: inline-block; animation: bounce 10s infinite;">
                <h1>{message}</h1>
            </div>
        </div>
        <style>
        @keyframes bounce {{
            0% {{ transform: translateX(0); }}
            50% {{ transform: translateX(calc(100% - 100vw)); }}
            100% {{ transform: translateX(0); }}
        }}
        </style>
        """
        st.markdown(html_message, unsafe_allow_html=True)


with st.sidebar:
    st.write("#### Select an image to upload.")
    uploaded_file_od = st.file_uploader("Image File Upload", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    col_bbox, col_blur, col_labels = st.columns(3)
    with col_bbox:
        show_bbox = st.radio("Show Bounding Boxes:", ["Yes", "No"], index=0, key="include_bbox")
    with col_blur:
        amount_blur = st.radio("Amount of Blur:", ["Low", "High"], index=1, key="amount_blur")
    with col_labels:
        show_class_label = st.radio("Show Class Labels:", ["Show Labels", "Hide Labels"], index=0,
                                    key="show_class_label")


##### Process uploaded image and run inference.

if uploaded_file_od:
    # Extract necessary components from project URL
    extracted_url = project_url_od.split("roboflow.com/")[1]
    parts = extracted_url.split("/")
    workspace_id = parts[0]
    model_id = parts[1]
    version_number = parts[3] if "model" in extracted_url else parts[2]

    # User-selected image.
    image = Image.open(uploaded_file_od)
    uploaded_img = np.array(image)
    inferenced_img = uploaded_img.copy()
    run_inference(workspace_id, model_id, version_number, uploaded_img, inferenced_img)