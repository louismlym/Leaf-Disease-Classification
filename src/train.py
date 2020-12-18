import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import glob
import time
import utils
import numpy as np
import matplotlib.pyplot as plt
from model import CNN
from torchvision import datasets, transforms
try:
  # For 2.7
  import cPickle as pickle
except:
  # For 3.x
  import pickle

# Define constants
TRAIN_PATH = "../data/train"
VAL_PATH = "../data/valid"
TEST_PATH = "../data/test"
EXPERIMENT_VERSION = "final_model" # change this to start a new experiment
LOG_PATH = '../logs/' + EXPERIMENT_VERSION + "/"

# Define hyperparamters
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.9
USE_CUDA = True
PRINT_INTERVAL = 50
WEIGHT_DECAY = 0.0005

# Define Train and Test functions
def train(model, device, train_loader, optimizer, epoch, log_interval):
  model.train()
  losses = []
  for batch_idx, (data, label) in enumerate(train_loader):
    data, label = data.to(device), label.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = model.loss(output, label)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        time.ctime(time.time()),
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
  return np.mean(losses)

def test(model, device, test_loader, log_interval=None):
  model.eval()
  test_loss = 0
  correct = 0
  confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

  with torch.no_grad():
    for batch_idx, (data, label) in enumerate(test_loader):
      data, label = data.to(device), label.to(device)
      output = model(data)
      test_loss_on = model.loss(output, label, reduction='sum').item()
      test_loss += test_loss_on
      pred = output.max(1)[1]
      stacked = torch.stack((label,pred), dim=1)
      for i, p in enumerate(stacked):
        tl, pl = p.tolist()
        confusion_matrix[tl, pl] = confusion_matrix[tl, pl] + 1
      correct_mask = pred.eq(label.view_as(pred))
      num_correct = correct_mask.sum().item()
      correct += num_correct
      if log_interval is not None and batch_idx % log_interval == 0:
        print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          time.ctime(time.time()),
          batch_idx * len(data), len(test_loader.dataset),
          100. * batch_idx / len(test_loader), test_loss_on))

  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), test_accuracy))
  return test_loss, test_accuracy, confusion_matrix

# Import and transform dataset (data augmentations)
transform = transforms.Compose([transforms.Resize(128),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(90),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]) 
                               ])
transform_test = transforms.Compose([transforms.Resize(128),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                    ])
data_train = datasets.ImageFolder(TRAIN_PATH, transform=transform)
data_test = datasets.ImageFolder(VAL_PATH, transform=transform_test)

NUM_CLASSES = len(data_train.classes)

# Now the actual training code
use_cuda = USE_CUDA and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
print('num cpus:', multiprocessing.cpu_count())

kwargs = {'num_workers': multiprocessing.cpu_count(),
          'pin_memory': True} if use_cuda else {}

class_names = data_train.classes

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                           shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE,
                                          shuffle=False, **kwargs)

model = CNN(NUM_CLASSES, device).to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
start_epoch = model.load_last_model(LOG_PATH) + 1
# read log
if (os.path.exists(LOG_PATH + "log.pkl")):
  train_losses, test_losses, test_accuracies = pickle.load(open(LOG_PATH + "log.pkl", 'rb'))
else:
  train_losses, test_losses, test_accuracies = ([], [], [])

test_loss, test_accuracy, confusion_matrix = test(model, device, test_loader, log_interval=10)

test_losses.append((start_epoch, test_loss))
test_accuracies.append((start_epoch, test_accuracy))
epoch = start_epoch - 1

try:
  for epoch in range(start_epoch, EPOCHS + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
    test_loss, test_accuracy, confusion_matrix = test(model, device, test_loader)
    train_losses.append((epoch, train_loss))
    test_losses.append((epoch, test_loss))
    test_accuracies.append((epoch, test_accuracy))
    # write log
    if not os.path.exists(os.path.dirname(LOG_PATH + "log.pkl")):
      os.makedirs(os.path.dirname(LOG_PATH + "log.pkl"))
    pickle.dump((train_losses, test_losses, test_accuracies), open(LOG_PATH + "log.pkl", 'wb'))

    model.save_best_model(test_accuracy, LOG_PATH + '%03d.pt' % epoch)


except KeyboardInterrupt as ke:
  print('Interrupted')
except:
  import traceback
  traceback.print_exc()
finally:
  model.save_model(LOG_PATH + '%03d.pt' % epoch, 0)
  ep, val = zip(*train_losses)
  utils.plot(ep, val, 'Train loss', 'Epoch', 'Error')
  ep, val = zip(*test_losses)
  utils.plot(ep, val, 'Test loss', 'Epoch', 'Error')
  ep, val = zip(*test_accuracies)
  utils.plot(ep, val, 'Test accuracy', 'Epoch', 'Accuracy (percentage)')
  utils.plot_confusion_matrix(confusion_matrix, data_train.classes)