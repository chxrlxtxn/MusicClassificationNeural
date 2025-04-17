import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import librosa
import librosa.display
import librosa.feature
import requests

fn = "SopSax.Vib.pp.C6Eb6.aiff"
url = "http://theremin.music.uiowa.edu/sound files/MIS/Woodwinds/sopranosaxophone/"+fn

req = requests.get(url)
with open(fn, "wb") as file:        
    file.write(req.content) # write to file

y, sr = librosa.load(fn)

# play audio file using IPython.display
import IPython.display as ipd
ipd.Audio(y, rate=sr) # load a NumPy array

# Extracting features via mel frequency cepstral coefficients. MEL SPECTOGRAM
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Downloading Data
data_dir = 'instrument_dataset/'
Xtr = np.load(data_dir+'uiowa_train_data.npy')
ytr = np.load(data_dir+'uiowa_train_labels.npy')
Xts = np.load(data_dir+'uiowa_test_data.npy')
yts = np.load(data_dir+'uiowa_test_labels.npy')

# Printing number of training and test samples and their details
print('training samples:', Xtr.shape[0])
print('test samples:', Xts.shape[0])
print('num features:', Xtr.shape[1])
print('num classes:', np.unique(ytr).shape[0]) # number of intsruments

# Standardizing data. Scale the training and test matrices
Xtr_scale = (Xtr - Xtr.mean(axis=0)) / Xtr.std(axis=0)
Xts_scale = (Xts - Xts.mean(axis=0)) / Xts.std(axis=0)

# Creating DataLoaders
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

batch_size = 100

# Create a training/test dataset from the tensors
train_ds = TensorDataset(torch.Tensor(Xtr_scale), torch.Tensor(ytr))
test_ds = TensorDataset(torch.Tensor(Xts_scale), torch.Tensor(yts))

# Create a training/test data loader from datasets
train_loader = DataLoader(train_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)

import torch.nn as nn
# construct the model
nin = Xtr_scale.shape[1]
nout = np.unique(ytr).shape[0]
nh = 256

# Create Net class
# nin: dimension of input data
# nh: number of hidden units
# nout: number of outputs
class Net(nn.Module):

# Initialize network
    def __init__(self, nin, nh, nout):
        super(Net,self).__init__()
        self.activation = nn.Sigmoid()
        self.Dense1 = nn.Linear(nin,nh)
        self.Dense2 = nn.Linear(nh,nout)

    def forward(self, x):
        x = self.activation(self.Dense1(x))
        out = self.activation(self.Dense2(x))
        return out
    
model = Net(nin=nin, nh=nh, nout=nout)

# Print string representation
print(str(model))

# Training the Network
import torch.optim as optim
lr = 1e-3
opt = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Train model for 10 epochs
num_epoch = 10

a_tr_loss = np.zeros([num_epoch])
a_tr_accuracy = np.zeros([num_epoch])
a_ts_loss = np.zeros([num_epoch])
a_ts_accuracy = np.zeros([num_epoch])

for epoch in range(num_epoch):

    model.train() # put model in training mode
    correct = 0 # initialize error counter
    total = 0 # initialize total counter
    batch_loss_tr = []
    # iterate over training set
    for train_iter, data in enumerate(train_loader):
        x_batch,y_batch = data
        y_batch = y_batch.type(torch.long)
        out = model(x_batch)
        # Compute Loss
        loss = criterion(out,y_batch)
        batch_loss_tr.append(loss.item())
        # Compute gradients using back propagation
        opt.zero_grad()
        loss.backward()
        # Take an optimization 'step'
        opt.step()
        # Do hard classification: index of largest score
        _, predicted = torch.max(out.data, 1)
        # Compute number of decision errors
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
    a_tr_loss[epoch] = np.mean(batch_loss_tr) # Compute average loss over epoch
    a_tr_accuracy[epoch] = 100*correct/total

    model.eval() # put model in evaluation mode
    correct = 0 # initialize error counter
    total = 0 # initialize total counter
    batch_loss_ts = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels = labels.type(torch.long)
            outputs = model(images)
            batch_loss_ts.append(criterion(outputs,labels).item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    a_ts_loss[epoch] = np.mean(batch_loss_ts)
    a_ts_accuracy[epoch] = 100*correct/total
    # Print details every print_mod epoch
    print('Epoch: {0:2d}   Train Loss: {1:.3f}   '.format(epoch+1, a_tr_loss[epoch])
          +'Train Acc: {0:.2f}    Test Loss: {1:.3f}   '.format(a_tr_accuracy[epoch], a_ts_loss[epoch])
          +'Test Acc: {0:.2f}'.format(a_ts_accuracy[epoch]))
    
# Plot the test accuract vs epoch
plt.plot(a_tr_accuracy)
plt.plot(a_ts_accuracy) # not getting 99%
plt.grid()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['training accuracy', 'test accuracy'])

plt.plot(a_tr_loss)
plt.semilogy()
plt.xlabel('batch')
plt.ylabel('loss')

# Optimizing Learning Rate
rates = [0.01,0.001,0.0001]
loss_hist = []
val_acc_hist = []

for rate in rates:
    opt = optim.Adam(model.parameters(), lr=rate)
    criterion = nn.CrossEntropyLoss()
    num_epoch = 10

    a_tr_loss = np.zeros([num_epoch])
    a_tr_accuracy = np.zeros([num_epoch])
    a_ts_loss = np.zeros([num_epoch])
    a_ts_accuracy = np.zeros([num_epoch])

    for epoch in range(num_epoch):

        model.train() # put model in training mode
        correct = 0 # initialize error counter
        total = 0 # initialize total counter
        batch_loss_tr = []
        # iterate over training set
        for train_iter, data in enumerate(train_loader):
            x_batch,y_batch = data
            y_batch = y_batch.type(torch.long)
            out = model(x_batch)
            # Compute Loss
            loss = criterion(out,y_batch)
            batch_loss_tr.append(loss.item())
            # Compute gradients using back propagation
            opt.zero_grad()
            loss.backward()
            # Take an optimization 'step'
            opt.step()
            # Do hard classification: index of largest score
            _, predicted = torch.max(out.data, 1)
            # Compute number of decision errors
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        a_tr_loss[epoch] = np.mean(batch_loss_tr) # Compute average loss over epoch
        a_tr_accuracy[epoch] = 100*correct/total

        model.eval() # put model in evaluation mode
        correct = 0 # initialize error counter
        total = 0 # initialize total counter
        batch_loss_ts = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                labels = labels.type(torch.long)
                outputs = model(images)
                batch_loss_ts.append(criterion(outputs,labels).item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        a_ts_loss[epoch] = np.mean(batch_loss_ts)
        a_ts_accuracy[epoch] = 100*correct/total
        # Print details every print_mod epoch
        print('Epoch: {0:2d}   Train Loss: {1:.3f}   '.format(epoch+1, a_tr_loss[epoch])
            +'Train Acc: {0:.2f}    Test Loss: {1:.3f}   '.format(a_tr_accuracy[epoch], a_ts_loss[epoch])
            +'Test Acc: {0:.2f}'.format(a_ts_accuracy[epoch]))
        
        plt.plot(a_tr_loss)
        plt.semilogy()
        plt.xlabel('batch')
        plt.ylabel('loss')
        plt.title('training rate')

        plt.legend(['0.01', '0.001', '0.0001'])
