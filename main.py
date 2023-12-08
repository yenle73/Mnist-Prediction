import pickle as pkl
from PIL import Image

st.title('MNIST Prediction')

def load_image(image_file):
	img = Image.open(image_file)
	return img

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
  image = load_image(image_file)
  st.image(image)
