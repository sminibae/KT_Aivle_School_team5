# imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from contextlib import redirect_stdout
from tqdm import tqdm


import os
import h5py
import json
import gc
import io

from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data

# Paths & indices
h5_path = '/home/all/processed_data/image_torchtensor_1024.h5'  # Update with your path
styles = np.load('/home/all/processed_data/styles_1024.npy', allow_pickle=True)  # Your styles data


# Load one image to get the input shape
with h5py.File(h5_path, 'r') as h5file:
    one_file = h5file['images'][0:1]  # Load the first image

# Not going to load X yet. because it is too big.
# We are going to load X batch by batch when model.fit.

le = LabelEncoder()
y = le.fit_transform(styles)

# Convert the NumPy array of labels into a torch tensor
# y_tensor = torch.from_numpy(y).long()  # Ensure it's a LongTensor for classification tasks


# Assuming total number of images
num_images = len(y)  # or len(combined_df)
indices = np.arange(num_images)

print('Data load success')

# Split indices
indices_train, indices_temp, y_train, y_temp = train_test_split(indices,y, test_size=0.2, random_state=1, stratify=y)
indices_val, indices_test, y_val, y_test = train_test_split(indices_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

np.save('indices_test.npy', np.array(indices_test))

class H5Dataset(Dataset):
    def __init__(self, h5_path, indices, styles):
        self.h5_path = h5_path
        self.indices = indices
        self.styles = styles

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5file:
            # Use the index to access the image and label
            image = h5file['images'][self.indices[idx]]
            styles = self.styles[self.indices[idx]]
            return torch.from_numpy(image).float(), torch.tensor(styles).long()

# Load your data and labels
train_data = H5Dataset(h5_path, indices_train, y)
val_data = H5Dataset(h5_path, indices_val, y)
test_data = H5Dataset(h5_path, indices_test, y)

batch_size = 16  # Define your batch size

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print('Data loader set')

# class for Offset. making offset as a trainable variable, and keep the already trained offset consistent within next batch.
class OffsetPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(OffsetPredictor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

# Handmade Conv2D Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.offset_predictor1 = OffsetPredictor(one_file.shape[1], 2*4*4, kernel_size=4, stride=1, padding=1)  # For a 4x4 kernel
        self.offset_predictor2 = OffsetPredictor(128, 2*4*4, kernel_size=4, stride=1, padding=1)
        self.offset_predictor3 = OffsetPredictor(96, 2*4*4, kernel_size=4, stride=1, padding=1)
        self.offset_predictor4 = OffsetPredictor(64, 2*4*4, kernel_size=4, stride=1, padding=1)
        self.offset_predictor5 = OffsetPredictor(32, 2*4*4, kernel_size=4, stride=1, padding=1)
        
        # Deformable Convolution layers
        
        # 3 -> 128
        self.deform_conv2d1 = DeformConv2d(in_channels=one_file.shape[1], out_channels=128, kernel_size=4, stride=1, padding=1)  # padding=1 for 'same'
        self.batchnorm2d1   = nn.BatchNorm2d(128)
        self.leakyrelu1     = nn.LeakyReLU(0.01)
        self.dropout2d1     = nn.Dropout2d(p=0.2)
        self.avgpool2d1     = nn.AvgPool2d(kernel_size=2)
        
        # 128 -> 96
        self.deform_conv2d2 = DeformConv2d(in_channels=128, out_channels=96, kernel_size=4, stride=1, padding=1)
        self.batchnorm2d2   = nn.BatchNorm2d(96)
        self.leakyrelu2     = nn.LeakyReLU(0.01)
        self.dropout2d2     = nn.Dropout2d(p=0.2)
        self.avgpool2d2     = nn.AvgPool2d(kernel_size=2)

        # 96 -> 64
        self.deform_conv2d3 = DeformConv2d(in_channels=96, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.batchnorm2d3   = nn.BatchNorm2d(64)
        self.leakyrelu3     = nn.LeakyReLU(0.01)
        self.dropout2d3     = nn.Dropout2d(p=0.2)
        self.avgpool2d3     = nn.AvgPool2d(kernel_size=2)
        
        # 64 -> 32
        self.deform_conv2d4 = DeformConv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.batchnorm2d4   = nn.BatchNorm2d(32)
        self.leakyrelu4     = nn.LeakyReLU(0.01)
        self.dropout2d4     = nn.Dropout2d(p=0.2)
        self.avgpool2d4     = nn.AvgPool2d(kernel_size=2)

        # 32 -> 16
        self.deform_conv2d5 = DeformConv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=1)
        self.batchnorm2d5   = nn.BatchNorm2d(16)
        self.leakyrelu5     = nn.LeakyReLU(0.01)
        self.dropout2d5     = nn.Dropout2d(p=0.2)
        self.avgpool2d5     = nn.AvgPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),  # 16384 개 나옴
            nn.Dropout(0.3),
            nn.Linear(15376,4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.01),
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 7),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Predict offsets for the first deformable convolution layer
        offset1 = self.offset_predictor1(x)        
        x = self.deform_conv2d1(x, offset1)
        x = self.batchnorm2d1(x)
        x = self.leakyrelu1(x)
        x = self.dropout2d1(x)
        x = self.avgpool2d1(x)
        
        offset2 = self.offset_predictor2(x)        
        x = self.deform_conv2d2(x, offset2)
        x = self.batchnorm2d2(x)
        x = self.leakyrelu2(x)
        x = self.dropout2d2(x)
        x = self.avgpool2d2(x)
        
        offset3 = self.offset_predictor3(x)        
        x = self.deform_conv2d3(x, offset3)
        x = self.batchnorm2d3(x)
        x = self.leakyrelu3(x)
        x = self.dropout2d3(x)
        x = self.avgpool2d3(x)
        
        offset4 = self.offset_predictor4(x)        
        x = self.deform_conv2d4(x, offset4)
        x = self.batchnorm2d4(x)
        x = self.leakyrelu4(x)
        x = self.dropout2d4(x)
        x = self.avgpool2d4(x)

        offset5 = self.offset_predictor5(x)        
        x = self.deform_conv2d5(x, offset5)
        x = self.batchnorm2d5(x)
        x = self.leakyrelu5(x)
        x = self.dropout2d5(x)
        x = self.avgpool2d5(x)        
        
        x = self.fc(x)
        return x

# Initialize the model
model = Model()

# Capture the summary output
summary_string = io.StringIO()
with redirect_stdout(summary_string):
    summary(model, input_size=one_file.shape[1:], device="cpu")
model_summary = summary_string.getvalue()
summary_string.close()

# Write the summary to a file
with open('model_summary.txt', 'w') as file:
    file.write(model_summary)
    
print('model_summary.txt saved')


# Setting device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move model to the chosen device
model = model.to(device)

# # Capture the summary output
# summary_string = io.StringIO()
# model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
# summary = summary_string.getvalue()
# summary_string.close()

# # Write the summary to a file
# with open('model_summary.txt', 'w') as file:
#     file.write(summary)


# Assuming 'model' is your PyTorch model
criterion = nn.CrossEntropyLoss()  # For categorical crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer
num_epochs = 10000000

# Early Stopping and Model Checkpoint can be manually implemented in the training loop
best_val_loss = float('inf')
patience = 10  # For early stopping
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=0.001)

# initialize history
history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}


print('start fitting')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Wrap your loader with tqdm for a progress bar
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for i, (inputs, labels) in pbar_train:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # for history
        # Calculate predictions for accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar_train.set_postfix({'loss': running_loss / (i + 1)})
        
    # Calculate average loss and accuracy over the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    
    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # Wrap your loader with tqdm for a progress bar
    pbar_eval = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    with torch.no_grad():
        for i, (inputs, labels) in pbar_eval:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # for history
            # Calculate predictions for accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar_eval.set_postfix({'loss': val_loss / (i + 1)})
            
    # Calculate average loss and accuracy over the validation set
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    
    # Append to history after each epoch
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    
    
    # for record in command prompt
    logs = f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n'
        
    print(logs)
    with open('logs.txt','a') as f:
        f.write(logs)
    
    
    # Reduce LR on plateau
    scheduler.step(val_loss)
    
    
    # Early stopping and Model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_state_dict.pth')
        torch.save(model, 'best_model.pth')
        patience = 5  # Reset patience since we found a better model
    else:
        patience -= 1
        if patience == 0:
            break
    
    # Garbage collection
    gc.collect()

    
print("Training complete")

with open('history.json', 'w') as f:
    json.dump(history, f)
print('Saved history.json')

# Save final model
torch.save(model.state_dict(), 'DeformConv2D_model_state_dict.pth')
torch.save(model, 'DeformConv2D_model.pth')
print('Saved model')

# history = model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=val_generator,
#     validation_steps=validation_steps,
#     epochs=1000000,  # Set the number of epochs
#     verbose=1,
#     callbacks=[early_stopping, reduce_lr, model_checkpoint, gc_callback],
# )


# model.save('Conv2D_handmade_model.h5')

# # Convert the history.history dict to a JSON file
# with open('history.json', 'w') as f:
#     json.dump(history.history, f)
