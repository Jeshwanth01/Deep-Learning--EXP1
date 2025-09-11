# EXPERIMENT 01: DEVELOPING A NEURAL NETWORK REGRESSION MODEL
## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Neural network regression models learn complex relationships between input variables and continuous outputs through interconnected layers of neurons. By iteratively adjusting parameters via forward and backpropagation, they minimize prediction errors. Their effectiveness hinges on architecture design, regularization, and hyperparameter tuning to prevent overfitting and optimize performance.
### Architecture:
  This neural network architecture comprises two hidden layers with ReLU activation functions, each having 5 and 3 neurons respectively, followed by a linear output layer with 1 neuron. The input shape is a single variable, and the network aims to learn and predict continuous outputs.

## NEURAL NETWORK MODEL:
<img width="1087" height="576" alt="Screenshot 2025-09-09 093958" src="https://github.com/user-attachments/assets/c992d847-9a33-4548-a500-f5d3e0574cf2" />

## DESIGN STEPS:
### STEP 1:
Loading the dataset.
### STEP 2:
Split the dataset into training and testing.
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.
## PROGRAM:
### Name: jeshwanth R
### Register Number: 2305003003
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['INPUT']].values
y = df[['OUTPUT']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)

```
## DATASET INFORMATION:
<img width="342" height="726" alt="Screenshot 2025-09-09 094007" src="https://github.com/user-attachments/assets/6a147f44-616e-4d79-b587-a0a29fca857d" />


## OUTPUT:
### Training Loss Vs Iteration Plot:
<img width="923" height="687" alt="Screenshot 2025-09-09 094015" src="https://github.com/user-attachments/assets/b6e1a083-9e82-437e-82ca-e503a528ad8a" />

### Epoch Training:
<img width="850" height="292" alt="Screenshot 2025-09-09 094020" src="https://github.com/user-attachments/assets/c33a0d7b-4936-4804-8a40-6ebcfbd2f743" />
### Test Data Root Mean Squared Error:
<img width="891" height="125" alt="Screenshot 2025-09-09 094025" src="https://github.com/user-attachments/assets/bd4dd8a1-105e-4a25-90fc-92a9ea3ebcf9" />

### New Sample Data Prediction:
<img width="654" height="93" alt="Screenshot 2025-09-09 094029" src="https://github.com/user-attachments/assets/e8b218e0-d2c8-4966-a48a-70df0a7b601b" />


## RESULT:
Thus a basic neural network regression model for the given dataset is written and executed successfully.
