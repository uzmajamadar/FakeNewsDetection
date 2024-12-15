import pandas as pd
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO

# Image preprocessing and model setup
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer
resnet.eval()

# Custom Dataset for loading images
class FakeNewsImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_url = row['Image']
        
        # Handle image download and processing
        try:
            response = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Could not download or process image: {image_url}, error: {e}")
            img = torch.zeros((3, image_size, image_size))  # Use a placeholder image
        
        return img

# Load datasets
train_df = pd.read_csv("train_IFND.csv")
val_df = pd.read_csv("val_IFND.csv")
test_df = pd.read_csv("test_IFND.csv")

# Create DataLoader
batch_size = 32

train_image_dataset = FakeNewsImageDataset(train_df, transform)
val_image_dataset = FakeNewsImageDataset(val_df, transform)
test_image_dataset = FakeNewsImageDataset(test_df, transform)

train_image_loader = DataLoader(train_image_dataset, batch_size=batch_size, shuffle=True)
val_image_loader = DataLoader(val_image_dataset, batch_size=batch_size, shuffle=False)
test_image_loader = DataLoader(test_image_dataset, batch_size=batch_size, shuffle=False)

# Function to extract image features using ResNet
def extract_image_features(loader, model, device):
    model.to(device)
    model.eval()
    image_features = []
    
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            features = model(images)
            features = features.view(features.size(0), -1)
            image_features.append(features.cpu().numpy())
    
    return np.vstack(image_features)

# Extract features for train, val, and test datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_image_features = extract_image_features(train_image_loader, resnet, device)
val_image_features = extract_image_features(val_image_loader, resnet, device)
test_image_features = extract_image_features(test_image_loader, resnet, device)

# Save or use the extracted features as needed
np.save('train_image_features.npy', train_image_features)
np.save('val_image_features.npy', val_image_features)
np.save('test_image_features.npy', test_image_features)

# Print shapes of the extracted features
print(f"Train image features shape: {train_image_features.shape}")
print(f"Val image features shape: {val_image_features.shape}")
print(f"Test image features shape: {test_image_features.shape}")
