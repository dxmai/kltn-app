import os

import numpy as np
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from helper import *

st.set_page_config(page_title='Image Search', page_icon=':mag_right:')
st.title("WaW")
st.write('# Who is that famous person & \n# What\'s the event?')

st.sidebar.write(":open_file_folder: Upload file")
uploaded_image = st.sidebar.file_uploader("", label_visibility="collapsed")
st.sidebar.write("or")
url = st.sidebar.text_input(":globe_with_meridians: Paste URL", '')

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

if show:
    st.sidebar.write("Preview image")
    st.sidebar.image(img)

run = st.sidebar.button("Search")
if run and img != '': 
    st.subheader("Predict")
    res_face, embedding = detect_face_ins(img)
    st.header(len(res_face))
    for face in res_face:
        each = get_roi(img, face)
        st.image(each, output_format="JPEG")
    st.image(img, output_format="JPEG")
else:
    st.write("")

path = os.getcwd() 

st.subheader("Sample search")
col1, col2, col3 = st.columns(3)
with col1:
    path1 = path + '/images/1.jpg'
    img1 = Image.open(path1)
    img1 = img1.resize((250, 250))
    st.image(img1)
    st.write(":adult: Vladimir Putin, Xi Jinping")
    st.write(":date: 4/2/2022")
    st.write(":ballot_box_with_check: Olympic Opening Ceremony")
    
with col2:
    path2 = path + '/images/2.jpg'
    img2 = Image.open(path2)
    img2 = img2.resize((250, 250))
    st.image(img2)
    st.write(":adult: Donal Trump")
    st.write(":date: 3/9/2022")
    st.write(":ballot_box_with_check: A Speech in Pennyslvania")

with col3:
    path3 = path + '/images/3.jpg'
    img3 = Image.open(path3)
    img3 = img3.resize((250, 250))
    st.image(img3)
    st.write(":adult: Joe Biden")
    st.write(":date: 4/7/2022")
    st.write(":ballot_box_with_check: Watching Fireworks at White House, US Independence Day")
