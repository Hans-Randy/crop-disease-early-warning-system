import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import kagglehub

# --- CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
# Detect if we have a GPU available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")
    
    # --- DATA DOWNLOAD ---
    print("Downloading dataset from Kaggle...")
    # This downloads the dataset and returns the path to where it was saved
    try:
        path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
        print(f"Dataset downloaded to: {path}")
            
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return

    # DATA PREPARATION
    # We use transforms to resize images and augment training data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Define the paths to the train and val directories
    train_dir = os.path.join(path, 'train')
    val_dir = os.path.join(path, 'val')

    # Load the dataset from the folder structure
    # ImageFolder expects folders to be named after the classes
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    except FileNotFoundError:
        print(f"Error: Could not find directory {path}. Please check your path.")
        return

    # Create DataLoaders
    # Note: We don't need random_split anymore because the folders are already split!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # MODEL SETUP (Transfer Learning)
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights='DEFAULT')

    # Freeze all layers so we don't retrain the whole thing
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier layer
    # MobileNetV2 classifier is a Sequential block; index 1 is the Linear layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))

    model = model.to(DEVICE)

    # TRAINING SETUP
    criterion = nn.CrossEntropyLoss()
    # We only optimize the parameters of the classifier head (the part we didn't freeze)
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE)

    # TRAINING LOOP
    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- NEW: Print start of epoch ---
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train() 
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # We use enumerate so we can track which batch we are on
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # --- NEW: Print progress every 20 batches ---
            if (batch_idx + 1) % 20 == 0:
                print(f"   > Processing batch {batch_idx + 1} / {len(train_loader)} ...")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train

        # Validation phase
        print("   > Running validation...")
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val

        print(f"RESULT: Epoch {epoch+1} completed | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
    # SAVE THE MODEL
    torch.save(model.state_dict(), 'plant_disease_model.pth')
    print("Model saved to plant_disease_model.pth")

if __name__ == '__main__':
    main()