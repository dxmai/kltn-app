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
    show = 1
elif uploaded_image:
    img = Image.open(BytesIO(uploaded_image.getvalue()))
    show = 1

if show:
    st.sidebar.write("Preview image")
    st.sidebar.image(img)

run = st.sidebar.button("Search")
if run and img != '': 
    res_face, embedding = detect_face_ins(img)
    st.header(len(res_face))
    for face in res_face:
        st.image(face)
    st.write("Run")
else:
    st.write("No")
