import torch
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import io

import R as arch

environment = 'cpu' 
device = torch.device(environment)

model_name = 'RRDB_ESRGAN_x4.pth'
models_dir = 'model/'

model_1 = '{}RRDB_ESRGAN_x4.pth'.format(models_dir)

def load_model(model_name, model_dir, device):
    model_path = "{}{}".format(model_dir, model_name)
    model = arch.RRDBNet(3, 3, 64, 23, gc = 32)
    model.load_state_dict(torch.load(model_path), strict = True)
    model.eval()
    model = model.to(device)
    return model

model = load_model(model_name, models_dir, device)

def super_resolution(img, device, model):
  img = np.array(img)
  img = img * 1.0 / 255
  img = torch.from_numpy(np.transpose(img[:,:, [2,1,0]], (2,0,1))).float()
  LR = img.unsqueeze(0)
  LR = LR.to(device)
  with torch.no_grad():
    result = model(LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
  result = np.transpose(result[[2,1,0], :, :], (1,2,0))
  result = (result * 255.0).round()
  #cv2.imwrite('results/{:s}_sr.png'.format('base'), result)
  return result
st.write("""# WELCOME TO IMAGE ENHANCHMENT""")

# load Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def decode_image(byte_array):
    image = Image.open(io.BytesIO(byte_array))
    return image



if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #image = cv2.imdecode(file_bytes, 1)
    image = decode_image(file_bytes)
    uploaded_image = image
    if image.size[0]<1000 and image.size[1]<1000:
      if image.size[0]<1000 and image.size[1]<1000 and image.size[0]>1000 and image.size[1]>1000:
        image = image.resize((300,300))
      st.write("""# Uploaded Image""")
      st.image(uploaded_file, caption=' ', use_column_width=True)
      st.write("""### Uploaded Image Shape""",uploaded_image.size)
      
      # Showing something until image is processing
      text_placeholder = st.empty()
      text_placeholder.text("""Please wait. Image is processing...""")
      
      # call super_regulation function
      result_sr = super_resolution(image, device, model)
      
      
      if result_sr is not None:
        result_sr_normalized = result_sr / 255.0
        st.write("""# Enhanched Image""")
        st.image(result_sr_normalized, caption=' ', use_column_width=True)
        st.write("""### Enhanched Image Shape:""", result_sr.shape)
        
        # Remove waiting text
        text_placeholder.empty()
        # Ddownload button creation
        result_sr = np.array(result_sr, dtype=np.uint8)
        
        images = Image.fromarray(result_sr)
        buf = BytesIO()
        images.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        btn = st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="Image.png",
        mime="image/jpeg",
        )
    else:
      st.write("### Your image is already enghanced and size is ", uploaded_image.size,". Please, upload low quality image like less than (1000,1000) size")
