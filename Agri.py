import streamlit as st
import pickle
#import sklearn
import pandas as pd
import numpy as np
from PIL import Image


#from streamlit_option_menu import option_menu

Agri = pickle.load(open('https://github.com/Engineer-Aman/Agricultural-crop-Prediction/blob/main/Agri_model.sav', 'rb'))

# horizontal Menu
selected2 = option_menu(None, ["Home", "Crop Predictor", "About", 'Settings'], 
 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected2

if (selected2 == "Home"):
    st.markdown("<h1 style='text-align: center; color:black;'>Agricultural Crop Prediction</h1>", unsafe_allow_html=True)

    #st.title('Agricultural Crop Prediction')
    image = Image.open('F.jpg')
    st.image(image, caption='',width=715)
    st.write('''
    Earlier, crop cultivation was undertaken on the basis of farmers’ hands-on expertise. However, climate change has begun to affect crop yields badly. Consequently, farmers are unable to choose the right crop/s based on soil and environmental factors, and the process of manually predicting the choice of the right crop/s of land has, more often than not, resulted in failure. Accurate crop prediction results in increased crop production. This is where machine learning playing a crucial role in the area of crop prediction. Crop prediction depends on the soil, geographic and climatic attributes. Selecting appropriate attributes for the right crop/s is an intrinsic part of the prediction undertaken by feature selection techniques. In this work, a comparative study of various wrapper feature selection methods are carried out for crop prediction using classification techniques that suggest the suitable crop/s for land. The experimental results show the Recursive Feature Elimination technique with the Adaptive Bagging classifier outperforms the others.
    ''')


if (selected2 == "Crop Predictor"):
    st.markdown("<h1 style='text-align: center; color:black;'>Crop Predictor</h1>", unsafe_allow_html=True)
    #st.title("Crop Yield Predictor")

    Nitrogen	= st.number_input(label='Enter Ratio Of Nitrogen In Soil',step=1.,format="%.2f")

    Phosphorus  = st.number_input(label='Enter Ratio Of Phosphorus In Soil',step=1.,format="%.2f")

    Potassium  = st.number_input(label='Enter Ratio Of Potassium  In Soil',step=1.,format="%.2f")

    temperature = st.number_input(label='Enter Temperature in "Celsius"',step=1.,format="%.2f")

    humidity = st.number_input(label='Enter Relative Humidity in %',step=1.,format="%.2f")

    ph       = st.number_input(label='Enter Ph value for the Soil',step=1.,format="%.2f")

    rainfall = st.number_input(label="Enter Rain Fall in 'mm'",step=1.,format="%.2f")

    if st.button('Click to check the crop'):
        Crop = Agri.predict([[Nitrogen,Phosphorus,Potassium ,temperature,humidity,ph,rainfall]])
        st.subheader('Suitable Crop For Your Soil is:')
        list=['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
        for i in list:
            if i==Crop:
                st.header(i.upper())
            
                
                
            
if (selected2 == "About"):
    
    st.write( '''
             A few weeks ago, I saw farmers working hardly and devoutly to grow crops for us. I was happy to see them work so hard.' But I found out that even after so many struggles our farmers grow crops to some extent. They failed to maximize crop growth. It made me think about why farmers are not able to increase their productivity.' Then I started thinking about the reasons why our agronomist could not reach maximum productivity.' 

             
Many factors affect the productivity of crops but two factors that affect crops productivity directly are,
1.	Soil Conditions
2.	Climate Conditions

Keeping these factors in my mind, I tried to build a Machine Learning model that will take different Climate and Soil conditions and predict a crop that can make maximum growth in these certain conditions and I did. By using this model, agronomists will be able to know which crop is more productive.

                ''')
     
if (selected2 == "Settings"):
    st.write('Currntly seetings are not available for use.'
             
             'Shortly will release it.')

with st.sidebar:
    
    selected = option_menu('Others',
                          
                          ['ABSTRACT',
                           'MOTIVATION',
                           'OBJECTIVES'],
                          
                          menu_icon="cast",
                          )
    
    if(selected=='ABSTRACT'):
        st.write('''
              As an agricultural country, India’s economy is predominantly depended over production of yield from
agriculture. About half of the population of India depends on agriculture for its livelihood, but its contribution
towards the GDP of India is only 14 per cent. One possible reason for this is the lack of adequate crop planning
by farmers. There is no system in place to advice farmers what crops to grow. In our study we found out that if
farmers knew the yield of the crop that they are planting beforehand, they would choose the crop that will
produce better yield based on the climate of that region. Knowing the crop yield before the harvest will also
help farmers and policy makers to make important decisions. The application that is developed in our research
will help the user to predict the crop yield based on the climatic parameters, previous yield and soil attributes.
Using appropriate machine learning techniques, the model learns the correlation between the yield and
features like soil type, rainfall. The predictions can be useful for industries in the agricultural sector and
farmers for proper choice of crops etc.
Agriculture is the best utility region especially inside the developing worldwide areas like India. Usage of records
age in agriculture can substitute the circumstance of decision making and Farmers can yield in higher manner.
About portion of the number of inhabitants in India relies upon on farming for its occupation however its
commitment towards the GDP of India is just 18 percent. One suitable explanation behind this is the deficiency of
adequate decision making by farmers on yield prediction. There isn’t any framework in location to suggest
farmer what plants to grow. The proposed machine learning approach aims at predicting the best yielded crop
for our country by analyzing various factors like rainfall, soil pH and past records of crops grown.
                ''')
        
    if(selected=='MOTIVATION'):  
        st.write('''
        Agriculture is the most important sector that influences the economy of India. It contributes to 18 percent of
India ‘s Gross Domestic Product. People of India are practicing Agriculture for years but the results are never
satisfying due to various factors that affect the crop yield. To fulfill the needs of around 1.2 billion people, it is
very important to have a good yield of crops. Due to factors like soil type, rainfall, temperature lack of technical
facilities etc. the crop yield is directly influenced. The system focuses on implementing crop yield prediction
system by using Machine learning techniques by doing analysis on agriculture dataset.

                  ''') 
        
    if(selected=='OBJECTIVES'): 
        st.write('''
        1. To use machine learning technique to predict crop yield.
2. To provide easy to use user interface and optimize agricultural production.
3. To increase the accuracy of crop yield prediction.
4. To analyze different parameters for predicting a crop
                  ''')
