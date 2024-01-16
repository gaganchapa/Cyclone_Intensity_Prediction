import streamlit as st
import cv2
from streamlit_lottie import st_lottie

from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import re
def extract_intensity(filename):
    match = re.search(r'_(\d+)', filename)
    return int(match.group(1)) if match else None
st.image("theindianwea.jpg")
st.title(":blue[CycloNet]")
st.caption("Experience the power of cutting-edge technology with our Cyclone Intensity Prediction web app.")
model = load_model("best_model.h5")
# st.header("Cyclone Intensity Predictior")
st.divider()
st.markdown("Simply upload a satellite image of the cyclone, and our advanced deep learning model will provide real-time predictions on the intensity.")
# import streamlit as st
on = st.toggle('Sample Predictions')
if on:
    st.image("savvv.png")
uploaded_file = st.file_uploader("Upload your file here...", type=['jpg',"png"])
if uploaded_file is not None:
    with st.spinner("Converting Image and predicting the Intensity"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        img=cv2.imdecode(file_bytes,1)
        st.image(img,caption="Uploaded Raw Image")
        infra = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        infra = cv2.applyColorMap(infra,cv2.COLORMAP_JET)
        infra_re = cv2.resize(infra,(512,512))
        infra_re = infra_re/255.0
        final = np.expand_dims(infra_re,axis=0)
        st.image(final,caption="pre-processed Image")
        pred = model.predict(final)[0][0]
        arg = np.argmax(pred)
        inte = pred[arg]
        # st.success(f"Predicted Intensity: {inte[0]}")
        plt.imshow(final[0])
        plt.axis("off")
        # plt.tight_layout()
        plt.title(f"Predicted Intensity: {inte[0]}")
        st.image("final.png")
        # cv2.imwrite("fin.png",)
        plt.savefig("final.png")
        org = st.toggle("Orginal Intensity")
        if org:
            match = re.search(r"name='([^']+)'", str(uploaded_file))
            if match:
                name_value = match.group(1)
                st.success(f"Original Intensity in Knots: {extract_intensity(name_value)}")
            else:
                st.exception("Couldn't Find the Original Intensity Try again:(")
            # st.markdown(extract_intensity(match))
        # st.markdown(uploaded_file)




