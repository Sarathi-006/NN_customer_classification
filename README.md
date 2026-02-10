# Developing a Neural Network Classification Model
## AIM
To develop a neural network classification model for the given dataset.
## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.
## Neural Network Model
![image](https://github.com/user-attachments/assets/51ec890b-fb80-42f1-9bcc-418a1cfda56d)
## DESIGN STEPS
### STEP 1:
Write your own steps
### STEP 2:
### STEP 3:
## PROGRAM
### Name: PARTHASARATHI S
### Register Number: 212223040144
```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
```
```python
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```
## Dataset Information
![image](https://github.com/user-attachments/assets/fd202505-2a62-46b7-8a6e-14fd57b2a6cb)

## OUTPUT
### Confusion Matrix
![image](https://github.com/user-attachments/assets/199baf24-362b-47b2-8fe1-25222df82a89)

### Classification Report
![image](https://github.com/user-attachments/assets/d3f7b504-ffd5-4511-9558-018c5f74a0d4)

### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/81f44d53-7845-4397-9c7f-fc5d92fa06c1)

## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
