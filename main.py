import numpy as np
import cv2
import pickle as pkl
import streamlit as st
from streamlit_drawable_canvas import st_canvas

input_md = open('lrc_mnist.pkl', 'rb')
model = pkl.load(input_md)

SIZE = 192
mode = st.checkbox("Draw or Delete?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

    rescaled = rescaled.reshape(1, -1) 

    if st.button('Predict'):
        rs = model.predict(rescaled)
        st.write(rs)

    
