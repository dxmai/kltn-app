import os

import numpy as np
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from helper import *
import matplotlib.pyplot as plt

# ====================== Header ======================
st.set_page_config(page_title='Image Search', page_icon=':mag_right:')
st.title("WaW")
st.write('# Who is that famous person & \n# What\'s the event?')

# ====================== Get input image ======================
st.sidebar.write(":open_file_folder: Tải ảnh lên")
uploaded_image = st.sidebar.file_uploader("uploader", label_visibility="collapsed")
st.sidebar.write("hoặc")
url = st.sidebar.text_input(":globe_with_meridians: Dán URL", '')

show = 0
img = ''
if url != '':
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    show = 1
elif uploaded_image:
    img = Image.open(BytesIO(uploaded_image.read()))
    img = np.array(img)
    show = 1

# ====================== Preview image ======================
if show:
    st.sidebar.write("Ảnh đã chọn")
    st.sidebar.image(img)

# ====================== Run model ======================
run = st.sidebar.button("Dự đoán")
if run and img != '': 
    st.subheader("Predict")
    res_face, embedding = detect_face_ins(img)
    # for face in res_face:
    #     each = get_roi(img, face)
    #     st.image(each, output_format="JPEG")
    # res_img = img
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_axes([0, 0, 1, 1])
    labels = draw_boundingbox(ax, res_face, ['test'] * len(res_face))
    string = ["{}: {}".format(key, value) for key, value in zip(labels.keys(), labels.values())]
    string = "; ".join(string)
    st.write(":adult:", string)
    plt.imshow(img)
    plt.axis('off')
    st.pyplot(fig)
else:
    st.write("")

# ====================== Sample Part ======================
path = os.getcwd() 
st.subheader("Một vài sự kiện mẫu")
col1, col2, col3  = st.columns(3)
with col1:
    path1 = path + '/images/1.jpg'
    img1 = Image.open(path1)
    img1 = img1.resize((300, 250), Image.Resampling.LANCZOS)
    st.image(img1, output_format="JPEG")
    st.write(":adult: Vladimir Putin, Xi Jinping")
    st.write(":date: 4/2/2022")
    st.write(":ballot_box_with_check: Lễ khai mạc Olympic")

with col2:
    path2 = path + '/images/2.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((300, 250))
    st.image(img2)
    st.write(":adult: Donal Trump")
    st.write(":date: 3/9/2022")
    st.write(":ballot_box_with_check: Một sự kiện ở Pennyslvania")

with col3:
    path3 = path + '/images/3.jpg'
    img3 = Image.open(path3)
    img3 = img3.resize((300, 250))
    st.image(img3)
    st.write(":adult: Joe Biden")
    st.write(":date: 25/8/2022")
    st.write(":ballot_box_with_check: Một sự kiện vận động Nước Mỹ an toàn hơn ở Maryland")
