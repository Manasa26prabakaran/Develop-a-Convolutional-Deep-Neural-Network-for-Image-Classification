# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
<img width="1241" height="212" alt="image" src="https://github.com/user-attachments/assets/12765302-8f81-4124-9a49-555b7a70b595" />

## Neural Network Model
<img width="1165" height="737" alt="dl" src="https://github.com/user-attachments/assets/14e736e0-8107-4a47-ac34-dc0ab4335b08" />

## DESIGN STEPS
### STEP 1:

Collect and preprocess the image dataset.

### STEP 2:

Import required deep learning libraries.

### STEP 3:

Build the CNN architecture.

### STEP 4:

Train the CNN model using training data.

### STEP 5:

Evaluate the model performance using test data.

### STEP 6:

Test the model with new images and verify predictions.
## PROGRAM

### Name:P Manasa

### Register Number:212224230149

```
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x
def train_model(model, train_loader, num_epochs=3):
   for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


      print('Name: P Manasa ')
      print('Register Number: 212224230149 ')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="482" height="325" alt="image" src="https://github.com/user-attachments/assets/7a5d885b-62f4-468d-b781-80ecea387751" />

## Confusion Matrix

<img width="1056" height="823" alt="image" src="https://github.com/user-attachments/assets/043587b4-f9db-4050-a789-984a9d0d109e" />

## Classification Report
<img width="643" height="458" alt="image" src="https://github.com/user-attachments/assets/fb1a4b18-02b5-492b-99ae-175d43c43afc" />

### New Sample Data Prediction

 <img width="603" height="656" alt="image" src="https://github.com/user-attachments/assets/ab3bfcf3-cb16-4a07-93fd-972adb8634ab" />
 
## RESULT
The CNN model was successfully trained and tested, achieving accurate image classification for new input images.
