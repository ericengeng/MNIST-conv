import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


class convNet(nn.Module):
  def __init__(self):
    super(convNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 64, 3, 1)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(2)
    self.conv2 = nn.Conv2d(64,64, 3,1)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(2)
    self.flatten = nn.Flatten()

    self.fc1 = nn.Linear(1600, 256)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(256, 10)

  def forward(self,x):
    x = self.relu1(self.conv1(x))
    x = self.maxpool1(x)
    x = self.relu2(self.conv2(x))
    x = self.maxpool2(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu3(x)
    x = self.fc2(x)
    return x
  
  def compute_loss_and_error(self, loader, criterion):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        self.eval() 
        with torch.no_grad():
            for data, label in loader:
                output = self.forward(data)
                loss = criterion(output, label)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_samples += label.size(0)
        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples
        error = 1 - accuracy
        return avg_loss, error
  

  def plot_metrics(self, train_metrics, val_metrics, label='Metric', filename='metrics_plot'):
    epochs = range(len(train_metrics))
    plt.figure(figsize=(10, 6))  # Optional: specify figure size
    plt.plot(epochs, train_metrics, label=f"Training {label}")
    plt.plot(epochs, val_metrics, label=f"Validation {label}")
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.title(f'Training and Validation {label}')  # Optional: add a title to the plot
    # Save the figure
    plt.savefig(f'{filename}_{label}.png')
    plt.close()  # Close the figure to free up memory



  def trainingLoop(self, train_loader, val_loader, optimizer, criterion, epochs=5):
    train_losses, train_errors, val_losses, val_errors = [], [], [], []
    
    # Measure before training starts
    train_loss, train_error = self.compute_loss_and_error(train_loader, criterion)
    val_loss, val_error = self.compute_loss_and_error(val_loader, criterion)
    train_losses.append(train_loss)
    train_errors.append(train_error)
    val_losses.append(val_loss)
    val_errors.append(val_error)
    print(f'Pre-training: Training Loss: {train_loss:.4f}, Training Error: {train_error:.4f}')
    print(f'Pre-training: Validation Loss: {val_loss:.4f}, Validation Error: {val_error:.4f}')
    start = time.time()
    for epoch in range(epochs):

        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        
        

        self.train()  
        for data, label in train_loader:
            optimizer.zero_grad()
            output = self.forward(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train_correct += (predicted == label).sum().item()
            total_train_samples += label.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_samples
        train_error = 1 - train_accuracy
        

        self.eval()  
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, label in val_loader:
                output = self.forward(data)
                loss = criterion(output, label)
                total_val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val_correct += (predicted == label).sum().item()
                total_val_samples += label.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_val_correct / total_val_samples
        val_error = 1 - val_accuracy

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Error: {train_error:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Error: {val_error:.4f}')
        train_losses.append(avg_train_loss)
        train_errors.append(train_error)
        val_losses.append(avg_val_loss)
        val_errors.append(val_error)


    end = time.time()
    total = end-start
    return train_losses, train_errors, val_losses, val_errors, total



  
if __name__ == '__main__':
  net = convNet()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum = 0.99)
  criterion = nn.CrossEntropyLoss()
  transform = torchvision.transforms.ToTensor()
  mnist_path = 'path/to/mnist/data'
  mnist_train = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)
  mnist_test = torchvision.datasets.MNIST(root=mnist_path, train=False, transform=transform)
  train_size = int(0.9*len(mnist_train))
  val_size = len(mnist_train) - train_size
  mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])
  train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
  val_loader = DataLoader(mnist_val, batch_size=32, shuffle=False)
  test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
  train_losses, train_errors, val_losses, val_errors, total_time =net.trainingLoop(train_loader, val_loader, optimizer, criterion, epochs=5)
  test_loss, test_error = net.compute_loss_and_error(test_loader, criterion)
  print(f'Test Loss: {test_loss:.4f}, Test Error: {test_error:.4f}')
  print(f"Total training time: {total_time} seconds")
  #net.plot_metrics(train_losses, val_losses, label='Loss', filename='adam')
  #net.plot_metrics(train_errors, val_errors, label='Errors', filename='adam')