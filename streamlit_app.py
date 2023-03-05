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
    model_path = os.getcwd() + '/model/knn_ver1.pickle'
    with open(model_path, "rb") as file:
        clf = pickle.load(file)
    dict_path = os.getcwd() + '/model/face_recognition_dict.pickle'
    with open(dict_path, "rb") as file:
        dic = pickle.load(file)
    event_path = os.getcwd() + '/even_info/labels_dict.pickle'
    with open(event_path, "rb") as file:
        event_dict = pickle.load(file)
    info_path = os.getcwd() + '/even_info/event_info_v1.pickle'
    with open(info_path, "rb") as file:
        event_info = pickle.load(file)
    event_model_p = os.getcwd() + '/model/knn_event_v1.pickle'
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
    st.write(img)
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
            if dic[name[0]] not in predicted:
                predicted.append(dic[name[0]])
        if len(predicted) != 0:
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
        # histogram_region = get_region_histogram(img)
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
        # ===== Histogram bin =====
        histogram_bin = CalHistogram(img)

        

        # ===== Classifier =====
        event_label = event_clf.predict(histogram_bin)
        st.write(event_label[0])
        st.write(event_dict)
        st.write(type(event_dict))
        st.write(event_dict[7])
        get_label = event_dict[event_label]
        st.write(get_label)
        for event in event_info:
            if event['id'] == get_label:
                info = event


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
    st.write(":adult: Kishida Fumio")
    st.write(":date: 15/11/2022")
    st.write(":ballot_box_with_check: Hội nghị G20")

with col2:
    path2 = path + '/images/2.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((300, 250))
    st.image(img2)
    st.write(":adult: Joe Biden")
    st.write(":date: 18/10/2022")
    st.write(":ballot_box_with_check: Sự kiện của Uỷ ban Quốc gia Đảng Dân chủ")

with col3:
    path3 = path + '/images/3.jpg'
    img3 = Image.open(path3)
    img3 = img3.resize((300, 250))
    st.image(img3)
    st.write(":adult: Vladimir Putin")
    st.write(":date: 7/9/2022")
    st.write(":ballot_box_with_check: Diễn đàn Kinh tế Phương Đông")
