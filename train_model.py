import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import time
from ResidualNetwork import RMN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    transform = transforms.Compose([
        transforms.Grayscale(),  
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    
    train_data_dir = 'images/train'
    val_data_dir = 'images/validation'

    train_data = datasets.ImageFolder(root=train_data_dir, transform=transform)
    val_data = datasets.ImageFolder(root=val_data_dir, transform=transform)

    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

    model = RMN().to(device)
    print(f"Model device: {next(model.parameters()).device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print("Train finished for epoch %d" % (epoch + 1))
        print("Epoch %d execution time: %s seconds" % (epoch + 1, time.time() - start))
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    start = time.time()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(all_labels, all_preds)
    print("Validation finished!")
    print("Validation execution time: %s seconds" % (time.time() - start))
    print(f"Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {val_accuracy}")
    
    torch.save(model.state_dict(), 'ResidualNet.pth')
    print("Model saved to ResidualNet.pth")

if __name__ == '__main__':
    start_main = time.time()
    main()
    print("Total execution time: %s seconds" % (time.time() - start_main))



