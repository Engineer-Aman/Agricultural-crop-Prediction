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

Nitrogen	Phosphorus	Potassium	temperature	humidity	ph	        rainfall	label
	90	42                    43	20.879744	82.002744	6.502985	202.935536	rice
	85	58	              41	21.770462	80.319644	7.038096	226.655537	rice
	60	55	              44	23.004459	82.320763	7.840207	263.964248	rice
	74	35	              40	26.491096	80.158363	6.980401	242.864034	rice
	78	42	              42	20.130175	81.604873	7.628473	262.717340	rice


df.shape

(2200, 8)
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
>**Performed Some EDA**

```
#lets count the crops values present in data
df['label'].value_counts()

Output:-
rice           100
maize          100
jute           100
cotton         100
coconut        100
papaya         100
orange         100
apple          100
muskmelon      100
watermelon     100
grapes         100
mango          100
banana         100
pomegranate    100
lentil         100
blackgram      100
mungbean       100
mothbeans      100
pigeonpeas     100
kidneybeans    100
chickpea       100
coffee         100

```

```
@interact
def summary(conditions=['Nitrogen','Phosphorus','Potassium','temperature',
                        'humidity','ph','rainfall']):
    print(df.groupby(['label']).agg({conditions : ['mean','max','min']}))
    
 
```
![1](https://user-images.githubusercontent.com/126685886/230737182-8892b9e5-3fe0-4224-9c7e-146a19c0c76e.png)

```
@interact
def summary(conditions=['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                        'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']):
    a=df[df['label']==conditions]
    data={'min':a.min(),'max':a.max(),'mean':a.mean()}
    d = pd.DataFrame(data)
    print(d)
    
```
![2](https://user-images.githubusercontent.com/126685886/230737230-b5c1ca5c-2b57-40c2-8051-5a6b98509663.png)

```
print("Summer Crops")
print(df[(df['temperature']>30)&(df['humidity']>50)]['label'].unique())
print("----------------------")
print("winter Crops")
print(df[(df['temperature']<20)&(df['humidity']>30)]['label'].unique())
print("----------------------")
print("Rainy Crops")
print(df[(df['rainfall']>200)&(df['humidity']>30)]['label'].unique())

Output:-

Summer Crops
['pigeonpeas' 'mothbeans' 'blackgram' 'mango' 'grapes' 'orange' 'papaya']
----------------------
winter Crops
['maize' 'pigeonpeas' 'lentil' 'pomegranate' 'grapes' 'orange']
----------------------
Rainy Crops
['rice' 'papaya' 'coconut']
```

```
print('crops which requires very high ratio of Nitrogen content in soil:', df[df['Nitrogen']>120]['label'].unique())
print('crops which requires very high ratio of Phosphorous content in soil:', df[df['Phosphorus']>100]['label'].unique())
print('crops which requires very high ratio of Potassium content in soil:', df[df['Potassium']>200]['label'].unique())
print('crops which requires very high rainfall:', df[df['rainfall']>200]['label'].unique())
print('crops which requires very low temperature:', df[df['temperature']<10]['label'].unique())
print('crops which requires very high temperature:', df[df['temperature']>40]['label'].unique())
print('crops which requires very low humidity:', df[df['humidity']<20]['label'].unique())
print('crops which requires very low ph:', df[df['ph']<4]['label'].unique())
print('crops which requires very high ph:', df[df['ph']>9]['label'].unique())


Output:-

crops which requires very high ratio of Nitrogen content in soil: ['cotton']
crops which requires very high ratio of Phosphorous content in soil: ['grapes' 'apple']
crops which requires very high ratio of Potassium content in soil: ['grapes' 'apple']
crops which requires very high rainfall: ['rice' 'papaya' 'coconut']
crops which requires very low temperature: ['grapes']
crops which requires very high temperature: ['grapes' 'papaya']
crops which requires very low humidity: ['chickpea' 'kidneybeans']
crops which requires very low ph: ['mothbeans']
crops which requires very high ph: ['mothbeans']
```
>**Distribution Graph**

![3](https://user-images.githubusercontent.com/126685886/230737382-8af08c72-e212-4498-8c65-94a87664ccc3.png)

we can see that some crops require a small amount of Nitrogen, some require average, and some require large amounts. Some crops require a small amount of Phosphorous, some require average, and some require large amounts. Some crops require a small amount of Potassium, some require average, and some require large amounts. Some crops require low Temperature, some require average, and some require high. Some crops require low Humidity in soil, some require average, and some require high. Some crops require low PH levels and some require high PH levels. Some crops require low Rain Fall, some require average, and some require high.

>**Splitting Data for Model Traning and testing**
```
x=df.drop(['label'],axis=1)
print(x.shape)
y=df['label']
print(y.shape)

Output:-
(2200, 7)
(2200,)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
```

>**Model Building**

```
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(x_train,y_train)

pred = log_model.predict(x_test)

```
