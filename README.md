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
<img width="1335" height="258" alt="Screenshot 2026-02-24 150711" src="https://github.com/user-attachments/assets/c142db8e-e22a-489c-b9b0-48c115d8073e" />


## OUTPUT
### Confusion Matrix
<img width="758" height="587" alt="Screenshot 2026-02-24 150622" src="https://github.com/user-attachments/assets/34a129f0-a380-4a80-87f4-bcdf6a4d4fe6" />

### Classification Report
<img width="1402" height="570" alt="Screenshot 2026-02-24 150811" src="https://github.com/user-attachments/assets/b153c8f9-223a-4d0f-9910-2eb1d3d22e48" />

### New Sample Data Prediction
<img width="769" height="102" alt="Screenshot 2026-02-24 150824" src="https://github.com/user-attachments/assets/de897a97-88e8-499a-b6d2-e270c432a6fd" />

## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
