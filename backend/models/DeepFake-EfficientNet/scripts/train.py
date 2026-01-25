#!/usr/bin/env python3
"""
Training script for DeepFake Detection
Robust training with checkpointing, logging, and monitoring.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

# Import from our package
from deepfake_detector.models import DeepFakeDetector
from deepfake_detector.data import DeepFakeDataset, get_train_transforms, get_val_transforms, create_dataloaders
from deepfake_detector.utils import setup_logger, calculate_metrics, plot_confusion_matrix, plot_training_history
from deepfake_detector.config import Config, load_config

from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Metrics
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, epoch, logger):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    conf_mat = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, conf_mat


def main():
    parser = argparse.ArgumentParser(
        description='Train DeepFake Detection Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--train-real', type=str, nargs='+', required=True,
                        help='Paths to training real images directories')
    parser.add_argument('--train-fake', type=str, nargs='+', required=True,
                        help='Paths to training fake images directories')
    parser.add_argument('--val-real', type=str, nargs='+', required=True,
                        help='Paths to validation real images directories')
    parser.add_argument('--val-fake', type=str, nargs='+', required=True,
                        help='Paths to validation fake images directories')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=8e-4,
                        help='Learning rate')
    parser.add_argument('--model', type=str, default='efficientnet-b1',
                        help='Model architecture')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Create output directories
    checkpoint_dir = Path(args.output_dir) / 'checkpoints'
    log_dir = Path(args.output_dir) / 'logs'
    results_dir = Path(args.output_dir) / 'results'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(
        name='training',
        log_file=str(log_dir / 'training.log'),
        level='INFO'
    )

    logger.info("="*60)
    logger.info("DeepFake Detection Training")
    logger.info("="*60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Creating datasets...")

    image_size = 240 if 'b1' in args.model else 224

    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)

    # Prepare data configs
    train_real_config = [(path, -1) for path in args.train_real]
    train_fake_config = [(path, -1) for path in args.train_fake]
    val_real_config = [(path, -1) for path in args.val_real]
    val_fake_config = [(path, -1) for path in args.val_fake]

    train_dataset = DeepFakeDataset(train_real_config, train_fake_config, train_transforms)
    val_dataset = DeepFakeDataset(val_real_config, val_fake_config, val_transforms)

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )

    # Create model
    logger.info(f"Creating model: {args.model}")
    model = DeepFakeDetector(model_name=args.model, pretrained=True)
    model = model.to(device)

    total_params, trainable_params = model.count_parameters()
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    num_train_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 5,
        num_training_steps=num_train_steps
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model.load_checkpoint(args.resume, device=str(device))
        # TODO: Load optimizer and scheduler state

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Training loop
    best_val_acc = 0.0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, logger
        )

        # Validate
        val_loss, val_acc, conf_mat = validate_epoch(
            model, val_loader, criterion, device, epoch, logger
        )

        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pth'
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            metrics={'val_acc': val_acc, 'val_loss': val_loss}
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = checkpoint_dir / 'best_model.pth'
            model.save_checkpoint(str(best_path), epoch=epoch)
            logger.info(f"Best model saved with val_acc: {best_val_acc:.4f}")

    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(
        history,
        metrics=['loss', 'accuracy'],
        save_path=str(results_dir / 'training_history.png'),
        show=False
    )

    logger.info("Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
