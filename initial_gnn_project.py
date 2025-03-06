import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#compose 2 transformations - 1. convert image to tensor 2. normalize pixel values to [-1, 1] to smooth out the model
transform = transforms.Compose([
    #transforms.Resize(16),  # Resize image to 16x16 instead of 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load dataset - mnist has predefined train and test sets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_subset = torch.utils.data.Subset(train_dataset, range(0, int(len(train_dataset) * 0.1)))  # 10% of training data
test_subset = torch.utils.data.Subset(test_dataset, range(0, int(len(test_dataset) * 0.1)))  # 10% of test data

# Use these subsets in the DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=64, shuffle=False)

# # Create DataLoader; batch_size=64; shuffling - always during training, not during testing
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: Compress; estract msot important features to make it cheaper
        self.encoder = nn.Sequential(
            #conv2d(input_channels, output_channels, kernel_size (filter), stride (n of pixel kernel moves at each step), padding (pixels added around the input))
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  #take max value in every 2x2 window, 

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
        )
        
        # Decoder: Decompress; reconstruct the image from the compressed version
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    total_loss = 0

    #z = model.encode(train_dataset) NO NEED FOR THIS! YOU ALREADY DID IT IN LINE 47
    for images, _ in train_loader:
        optimizer.zero_grad() #reset gradients
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass (encode -> decode)
        reconstructed = model(images)
            
        # Compute loss
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
            
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def test():
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    
    for images, _ in test_loader:  # No need for labels
        images = images.to(device)
        
        # Forward pass (encode -> decode)
        reconstructed = model(images)
        
        # Compute loss
        loss = criterion(reconstructed, images)
        total_loss += loss.item()
    
    return total_loss / len(test_loader)

for epoch in range(300):
    loss = train()
    tloss = test()
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>2} | Loss: {loss:.4f} | Tloss: {tloss:.4f}')
