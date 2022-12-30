import os

import numpy as np
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from helper import *
from feature_extractor import *
import matplotlib.pyplot as plt
import pickle 
import cv2
import subprocess

# ====================== Header ======================
st.set_page_config(page_title='Image Search', page_icon=':mag_right:')
# st.title("WaW")
st.write('# Who is that famous person & \n# What\'s the event?')

@st.cache
def install():
    subprocess.run(["python3", "-m", "pip", "install", "paddlepaddle"])
    subprocess.run(["python3", "-m", "pip", "install", "opencv-python"])
    subprocess.run(["python3", "-m", "pip", "install", "tqdm"])
    subprocess.run(["python3", "-m", "pip", "install", "pyyaml"])
    subprocess.run(["python3", "-m", "pip", "install", "sklearn==0.0"])

# install()

# ====================== Load additional ======================
def load():
    model_path = os.getcwd() + '/model/knn_insight.pickle'
    with open(model_path, "rb") as file:
        clf = pickle.load(file)
    dict_path = os.getcwd() + '/model/dict_insight.pickle'
    with open(dict_path, "rb") as file:
        dic = pickle.load(file)
    event_path = os.getcwd() + '/even_info/event_dict.pickle'
    with open(event_path, "rb") as file:
        event_dict = pickle.load(file)
    info_path = os.getcwd() + '/even_info/event_info.pickle'
    with open(info_path, "rb") as file:
        event_info = pickle.load(file)
    event_model_p = os.getcwd() + '/model/knn_event.pickle'
    with open(event_model_p, "rb") as file:
        event_clf = pickle.load(file)
    return clf, dic, event_dict, event_info, event_clf

clf, dic, event_dict, event_info, event_clf = load()

path = os.getcwd() 

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
# if show:
#     st.sidebar.write("Ảnh đã chọn")
#     st.sidebar.image(img)

# ====================== Run model ======================
# run = st.sidebar.button("Dự đoán")
if img != '': 
    with st.spinner("Vui lòng chờ một chút..."):
        # ====================== Face Dectection and Recognition ======================
        res_face, embeddings = detect_face_ins(img)
        fig = plt.figure(figsize = (5, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        predicted = []
        for embedding in embeddings:
            embedding = embedding.reshape(-1, 512)
            name = clf.predict(embedding)
            predicted.append(dic[name[0]])
        labels = draw_boundingbox(ax, res_face, predicted)

        # ====================== Matting ======================
        # Save image to get matting input
        # source_img = 'query.jpg'
        # cv2.imwrite(source_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 
        # Matting result image
        # des_img = path + "/result/matting.jpg"

        # subprocess.run(["python", "download_inference_models.py"])
        # subprocess.run(["python", "download_data.py"])
        # subprocess.run(["python", "seg_demo.py", "--config", "inference_models/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax/deploy.yaml", "--img_path", source_img, "--save_dir", des_img])
    
        # # Get the background
        # res = Image.open(des_img)
        # res = np.array(res)
        # background = np.where(res==254, img, 0)
        
        # ====================== Event Classification ======================

        # ===== Extract original image =====
        # Histogram region
        histogram_region = get_region_histogram(img)
        # st.write(histogram_region.shape)

        # # MobileNet
        # mobile_net = get_mobilenet(img)
        # mobile_net = mobile_net.reshape(mobile_net.shape[0], mobile_net.shape[1] * mobile_net.shape[2] * mobile_net.shape[3])
        # st.write(mobile_net.shape)

        # # ===== Extract background image =====
        # # Hisrogram RGB
        # hist_rgb = CalHistogram_RGB(background)
        # st.write(hist_rgb.shape)

        # # Dominant color
        # number_of_colors = 5
        # dominant_color = get_dominant_color(img, 5)
        # dominant_color = check_color(dominant_color)
        # st.write(dominant_color.shape)
        
        # # Concatenate
        # event_feature = np.concatenate((histogram_region, mobile_net, hist_rgb, dominant_color), axis=None)
        # event_feature = event_feature.reshape(1, -1)
        # st.write(event_feature.shape)

        # ===== Classifier =====
        event_label = event_clf.predict(histogram_region)
        info = event_info[int(event_label)]


    # col1, col2  = st.columns(2)
    # with col1:
    string = ["{}: {}".format(key, value) for key, value in zip(labels.keys(), labels.values())]
    string = "; ".join(string)
    st.write(":adult:", string)
    st.write(":date:", info['event_date'])
    st.write(":ballot_box_with_check:", info['event_name'])
    # with col2:
    plt.imshow(img)
    plt.axis('off')
    st.pyplot(fig)

# ====================== Sample Part ======================
st.subheader("Một vài sự kiện mẫu")
col1, col2, col3  = st.columns(3)
with col1:
    path1 = path + '/images/1.jpg'
    img1 = Image.open(path1)
    img1 = img1.resize((300, 250), Image.Resampling.LANCZOS)
    st.image(img1, output_format="JPEG")
    st.write(":adult: Vladimir Putin, Tập Cận Bình")
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
