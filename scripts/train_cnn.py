#!/usr/bin/env python3
"""
Train CNN model for MRI-based dementia prediction.

This script trains a convolutional neural network on MRI images
for dementia classification.

Usage:
    python train_cnn.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from cnn_model import SimpleMRI2DCNN


class MRIDataset(Dataset):
    """Dataset class for MRI images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def setup_logging(config):
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def collect_image_data(data_root, classes):
    """Collect image paths and labels."""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_root, class_name)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(class_idx)
    
    return image_paths, labels


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def main(config_path):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info("Starting CNN training pipeline")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    data_config = config['data']
    data_root = data_config['raw_dir']
    classes = ['Non Demented', 'Mild Dementia']
    
    logger.info("Collecting MRI image data...")
    image_paths, labels = collect_image_data(data_root, classes)
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        logger.error("No images found. Please ensure MRI data is downloaded and placed correctly.")
        logger.error("See data/README_data.md for instructions.")
        return
    
    # Define transforms
    cnn_config = config['models']['cnn']
    img_size = tuple(cnn_config['image_size'])
    
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Split data
    train_config = config['training']
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=train_config['test_size'],
        random_state=train_config['random_state'],
        stratify=labels
    )
    logger.info(f"Train set: {len(train_paths)}, Test set: {len(test_paths)}")
    
    # Create datasets and loaders
    train_dataset = MRIDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = MRIDataset(test_paths, test_labels, transform=test_transform)
    
    batch_size = cnn_config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    num_classes = cnn_config['num_classes']
    model = SimpleMRI2DCNN(num_classes=num_classes).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cnn_config['learning_rate'])
    
    # Training loop
    num_epochs = cnn_config['epochs']
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            models_dir = config['outputs']['models_dir']
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, 'cnn_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'classes': classes,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'epoch': epoch
            }, model_path)
            logger.info(f"Saved best model (acc: {best_acc:.2f}%) to {model_path}")
    
    logger.info(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN model for dementia prediction")
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)
