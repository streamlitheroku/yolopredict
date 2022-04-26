import os
import streamlit as st
from PIL import Image
import torch
import pandas as pd

st.set_page_config(page_title='YOLO Classifier', page_icon='favicon.png')
def load_image(image_file):
	img = Image.open(image_file)
	return img

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://www.pixelstalk.net/wp-content/uploads/images2/Download-Free-African-Animals-Wallpapers.png");
             background-size: cover;

         }}
         </style>
         """,
         unsafe_allow_html=True
     )

@st.cache(ttl=48*3600)
def load_model():
  model = torch.hub.load('yolov5','custom',path='best.pt',source='local', device='cpu',force_reload=True)
  return model
def predict():
   # giving a title
   st.subheader('Upload either Buffalo/Elephant/Rhino/Zebra image for prediction')
   image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
   # code for Prediction
   prediction = ''
   
           
        
        

   # creating a button for Prediction
   if st.button('Predict'):
     if image_file is not None:
         # To See details
        model = load_model()
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        st.write(file_details)
        img = load_image(image_file)
        st.image(img,width=640)
        # st.video(data, format="video/mp4", start_time=0)
        with st.spinner('Predicting...'):
            result=model(img,size=640)
            l= result.pandas().xyxy[0]['name']
        d={}
        for i in l:
          d[i]=d.get(i,0)+1
        s=""
        for i in d:
          s+=f"{d[i]} {i}, "
        st.success(s[:-2])
        st.image(Image.fromarray(result.render()[0]))

def home():
  st.write("## YOLO is used for Real-Time Object Detection")
  st.write("#### We passed this image for prediction through our YOLO model")
  st.image("original.jpg", width=800)
  st.write("#### This is the output, our model predicted")
  st.image("predictions_img.jpeg", width=800)
  st.write('## To know more about the project, visit the About page')

def about():
  st.write("# Team SMOTE with Real-Time Animal Detection using YOLO")
  st.write("### All the files for this project can be found on our Github Repository [here](https://github.com/heyakshayhere/Hamoye_capstone_project_smote/)")

def main():
  hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
	footer:after {
	content:'Team SMOTE'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
}
        </style>
        """
  st.markdown(hide_menu_style, unsafe_allow_html=True)
  st.write('<style>.css-10trblm{ text-align:center;} .css-12oz5g7 { max-width: 50rem;} div.row-widget.stRadio > div{flex-direction:row;justify-content: space-evenly;} div.row-widget.stRadio > div > label > div{font-size:24px !important;}</style>',unsafe_allow_html=True)
  st.title('YOLO Animal Classifier')
  page = st.radio("", ["Home","Predict","About"])
  # Website flow logic    
  if page == "Home":
      set_bg_hack_url()
      home()
  elif page == "Predict":
      predict()
  elif page == "About":
      set_bg_hack_url()
      about()

if __name__ == '__main__':
    main()

