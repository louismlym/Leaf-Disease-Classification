import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, num_classes, device):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.maxPool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(32, 64, 3)
    self.maxPool2 = nn.MaxPool2d(2)
    self.conv3 = nn.Conv2d(64, 128, 3)
    self.conv3_bn = nn.BatchNorm2d(128)
    self.maxPool3 = nn.MaxPool2d(2)
    self.fc1 = nn.Linear(25088, 256)
    self.fc1_bn = nn.BatchNorm1d(256)
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(256, num_classes)
    self.accuracy = None
    self.device = device
  
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.maxPool1(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.maxPool2(x)
    x = self.conv3(x)
    x = self.conv3_bn(x)
    x = F.relu(x)
    x = self.maxPool3(x)
    x = F.relu(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc1_bn(x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  
  def loss(self, prediction, label, reduction='mean'):
    loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
    return loss_val
  
  def save_model(self, file_path, num_to_keep=1):
    utils.save(self, file_path, num_to_keep)
  
  def save_best_model(self, accuracy, file_path, num_to_keep=1):
    if self.accuracy == None or accuracy > self.accuracy:
      self.accuracy = accuracy
      self.save_model(file_path, num_to_keep)
  
  def load_model(self, file_path):
    utils.restore(self, file_path, self.device)
  
  def load_last_model(self, dir_path):
    return utils.restore_latest(self, dir_path, self.device)