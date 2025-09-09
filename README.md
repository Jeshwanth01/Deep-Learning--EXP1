**Developing a Neural Network Regression Model**

**AIM**
To develop a neural network regression model for the given dataset.


**THEORY**

Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

**Neural Network Model**
<img width="1071" height="524" alt="Screenshot 2025-09-09 091231" src="https://github.com/user-attachments/assets/48c8c904-d6f1-4ccf-9c34-ac679174b732" />

**DESIGN STEPS**

**STEP 1: Generate Dataset**
Create input values from 1 to 50 and add random noise to introduce variations in output values .

**STEP 2: Initialize the Neural Network Model**
Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

**STEP 3: Define Loss Function and Optimizer**
Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

**STEP 4: Train the Model**
Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

**STEP 5: Plot the Loss Curve**
Track the loss function values across epochs to visualize convergence.

**STEP 6: Visualize the Best-Fit Line**
Plot the original dataset along with the learned linear model.

**STEP 7: Make Predictions**
Use the trained model to predict for a new input value .

**PROGRAM**

**Name**:Jeshwanth R

**Register Number:**2305003003


class Model(nn.Module):

    def __init__(self, in_features, out_features):
       
        super().__init__()
        
        #Include your code here




# Initialize the Model, Loss Function, and Optimizer


<img width="313" height="714" alt="Screenshot 2025-09-09 090714" src="https://github.com/user-attachments/assets/dd1e8df0-26fc-4a44-a9d3-72745d79d023" />



**OUTPUT**

<img width="927" height="731" alt="Screenshot 2025-09-09 090727" src="https://github.com/user-attachments/assets/9f520474-ff27-4479-924e-1c0945873c10" />

**New Sample Data Prediction**

<img width="866" height="282" alt="Screenshot 2025-09-09 090806" src="https://github.com/user-attachments/assets/607715c7-4a76-4d34-8574-1d1a1f980a42" />

<img width="880" height="111" alt="Screenshot 2025-09-09 090820" src="https://github.com/user-attachments/assets/d7612d9b-ce7d-4544-85d2-7d49b63c4bf7" />

<img width="862" height="99" alt="Screenshot 2025-09-09 090833" src="https://github.com/user-attachments/assets/01e3e749-896c-434a-87c0-3dff985f10da" />

**RESULT**

Thus, a neural network regression model was successfully developed and trained using PyTorch.
