# Agricultural-crop-Prediction

:wave: Hi in this data science project I am going to predict the 'Crop' based on soil proerties and climate.

In below video you can check the workflow and overview of this project.:point_down:

https://user-images.githubusercontent.com/126685886/230735981-8be7621f-7053-4534-ab8b-64f0d7a8b6f5.mp4

To access this webapp use below link:point_down:

https://engineer-aman-agricultural-crop-prediction-agri-model-ejr8j6.streamlit.app/



A few weeks ago, I saw farmers working hardly and devoutly to grow crops for us. I was happy to see them work so hard.
But I found out that even after so much struggle our farmers grow crops to some extent. They failed to maximize crop growth. It made me think about why farmers are not able to increase their productivity.
Then I started thinking about the reasons why our agronomist could not reach maximum productivity.


Many factors affect the productivity of crops but two factors that affect crops productivity directly are,

1.Soil Conditions
2.Climate Conditions

Keeping these factors in my mind, I tried to build a Machine Learning model that will take different Climate and Soil conditions and predict a crop that can make maximum growth in these certain conditions and I did. By using this model, agronomists will be able to know which crop is more productive.


>### Let me tell you what I did and what I learned in this project…

**Tools Used:-**

1.Jupyter notebook
2.Streamlit
3.Python

> **About Data Set**

I got the data Set from Kaggle to build such a model.

```
df.head()

Nitrogen	Phosphorus	Potassium	temperature	humidity	ph	rainfall	label
	90	42                    43	20.879744	82.002744	6.502985	202.935536	rice
	85	58	              41	21.770462	80.319644	7.038096	226.655537	rice
	60	55	              44	23.004459	82.320763	7.840207	263.964248	rice
	74	35	              40	26.491096	80.158363	6.980401	242.864034	rice
	78	42	              42	20.130175	81.604873	7.628473	262.717340	rice

```
Columns and there Descrption are as below :point_down: 

**N:** Ratio of Nitrogen in the Soil

**P:** Ratio of Phosphorous in the Soil

**K:** Ratio of Potassium in the Soil

**Temperature:** Temperature in Celsius

**Humidity:** Relative Humidity in %

**PH:** Ph value for the Soil

**Rain Fall:** Rain Fall in 1mm

>**Data Cleaning**

After getting the Data Set, the first step is cleaning data. Data cleaning is the process of preparing raw data for analysis by removing bad data, organizing the raw data, and filling in the null values. But we don’t have any need to clean Data because our data set is already cleaned. There are no missing values and unwanted columns or rows.

```
df.isnull().sum()

Nitrogen       0
Phosphorus     0
Potassium      0
temperature    0
humidity       0
ph             0
rainfall       0
label          0

```

