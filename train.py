#!/usr/bin/env python3
"""Training script for VGG-based image-to-text model for Kamon descriptions."""

import os
import sys
import json
import jsonlines
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from absl import app, flags

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

import kamon_dataset as kd
from vgg_image_to_text_model import VGGImageToTextModel

# Define command line flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 224, 'Input image size')
flags.DEFINE_integer('num_train_augmentations', 9, 'Number of training augmentations')
flags.DEFINE_integer('batch_size', 16, 'Batch size for training')
flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('checkpoint_steps', 10000, 'Steps between checkpoints')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Directory to save checkpoints')
flags.DEFINE_string('output_dir', 'outputs', 'Directory to save outputs')
flags.DEFINE_integer('max_checkpoints_to_keep', 5, 'Maximum number of checkpoints to keep')
flags.DEFINE_boolean('also_train_vgg', False, 'Whether to train VGG parameters')
flags.DEFINE_integer('ngram_length', 2, 'N-gram context length')
flags.DEFINE_integer('hidden_dim', 512, 'Hidden dimension for feature combiner')
flags.DEFINE_string('device', 'auto', 'Device to use (cuda, cpu, or auto)')


def manage_checkpoints(checkpoint_dir, max_to_keep):
    """Remove old checkpoints, keeping only the most recent max_to_keep."""
    if max_to_keep <= 0:
        return

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= max_to_keep:
        return

    # Sort by step number (extract from filename)
    def get_step_number(filepath):
        basename = os.path.basename(filepath)
        # Extract step number from "checkpoint_step_12345.pt"
        step_str = basename.replace("checkpoint_step_", "").replace(".pt", "")
        try:
            return int(step_str)
        except ValueError:
            return 0

    checkpoint_files.sort(key=get_step_number)

    # Remove oldest checkpoints
    files_to_remove = checkpoint_files[:-max_to_keep]
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed old checkpoint: {os.path.basename(file_path)}")
        except OSError as e:
            print(f"Warning: Could not remove checkpoint {file_path}: {e}")


def save_mask_images(masks, output_dir, prefix, vocab, label_to_expr):
    """Save mask images as PNG files.

    Args:
        masks: Tensor of shape [B, seq_len, H, W]
        output_dir: Directory to save images
        prefix: Prefix for image names (e.g., 'test_000')
        vocab: Vocabulary size
        label_to_expr: Dictionary mapping labels to expressions
    """
    batch_size, seq_len, height, width = masks.shape

    for batch_idx in range(batch_size):
        # Create directory for this example
        example_dir = os.path.join(output_dir, f"{prefix}_{batch_idx:03d}")
        os.makedirs(example_dir, exist_ok=True)

        for seq_idx in range(seq_len):
            # Convert mask to numpy array and scale to [0, 255]
            mask = masks[batch_idx, seq_idx].cpu().numpy()
            mask_img = (mask * 255).astype(np.uint8)

            # Save as PNG
            img = Image.fromarray(mask_img, mode='L')
            img_path = os.path.join(example_dir, f"img_{seq_idx:03d}.png")
            img.save(img_path)


def evaluate_model(model, test_loader, device, vocab_size, label_to_expr, end_token, output_dir, step):
    """Evaluate model on test set and save results."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_masks = []

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        for batch_idx, (images, target_tokens) in enumerate(test_loader):
            images = images.to(device)
            target_tokens = target_tokens.to(device)

            # Forward pass
            logits, masks = model(images, target_tokens)

            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_tokens.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Generate predictions
            pred_tokens, pred_masks = model.generate(images, end_token)

            # Convert predictions to text and include ground truth
            for i in range(batch_size):
                # Get predicted tokens
                pred_tokens_list = pred_tokens[i].cpu().tolist()
                # Find end token position in predictions
                try:
                    end_pos = pred_tokens_list.index(end_token)
                    pred_tokens_list = pred_tokens_list[:end_pos]
                except ValueError:
                    pass

                predicted_description = ' '.join([label_to_expr.get(token, f'<UNK:{token}>') for token in pred_tokens_list])

                # Get ground truth tokens
                gt_tokens_list = target_tokens[i].cpu().tolist()
                # Find end token position in ground truth
                try:
                    gt_end_pos = gt_tokens_list.index(end_token)
                    gt_tokens_list = gt_tokens_list[:gt_end_pos]
                except ValueError:
                    pass

                reference_description = ' '.join([label_to_expr.get(token, f'<UNK:{token}>') for token in gt_tokens_list])

                all_predictions.append({
                    'batch_idx': batch_idx,
                    'example_idx': i,
                    'predicted_description': predicted_description,
                    'predicted_tokens': pred_tokens_list,
                    'reference_description': reference_description,
                    'reference_tokens': gt_tokens_list
                })

            all_masks.append(pred_masks.cpu())

            # Save mask images for first few batches
            if batch_idx < 5:  # Save masks for first 5 batches
                save_mask_images(
                    pred_masks.cpu(),
                    os.path.join(output_dir, f'step_{step}'),
                    f'test_{batch_idx:03d}',
                    vocab_size,
                    label_to_expr
                )

    avg_loss = total_loss / total_samples

    # Save predictions to jsonlines file
    predictions_path = os.path.join(output_dir, f'step_{step}', 'predictions.jsonl')
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    with jsonlines.open(predictions_path, 'w') as writer:
        for pred in all_predictions:
            writer.write(pred)

    return avg_loss


def main(argv):
    del argv  # Unused

    # Set device
    if FLAGS.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(FLAGS.device)

    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_data = kd.KamonDataset(
        division="train",
        image_size=FLAGS.image_size,
        num_augmentations=FLAGS.num_train_augmentations,
        one_hot=False,  # We need integer labels, not one-hot
        omit_edo=True,
    )

    test_data = kd.KamonDataset(
        division="test",
        image_size=FLAGS.image_size,
        num_augmentations=1,
        one_hot=False,
        omit_edo=True,
    )

    # Extract vocabulary information
    vocab_size = train_data.vocab_size
    label_to_expr = train_data.label_to_expr
    end_token = train_data.end_token
    max_seq_len = train_data.max_len

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"End token ID: {end_token}")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print("Initializing model...")
    model = VGGImageToTextModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        image_size=FLAGS.image_size,
        ngram_length=FLAGS.ngram_length,
        hidden_dim=FLAGS.hidden_dim,
        also_train_vgg=FLAGS.also_train_vgg,
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Training loop
    print("Starting training...")
    global_step = 0

    for epoch in range(FLAGS.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (images, target_tokens) in enumerate(train_loader):
            images = images.to(device)
            target_tokens = target_tokens.to(device)

            # Forward pass
            logits, masks = model(images, target_tokens)

            # Calculate loss
            batch_size, seq_len, vocab_size_out = logits.shape
            logits_flat = logits.view(-1, vocab_size_out)
            targets_flat = target_tokens.view(-1)

            loss = criterion(logits_flat, targets_flat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Checkpoint and evaluation
            if global_step % FLAGS.checkpoint_steps == 0:
                print(f"\nCheckpointing at step {global_step}...")

                # Save checkpoint
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'vocab_size': vocab_size,
                    'max_seq_len': max_seq_len,
                    'end_token': end_token,
                    'label_to_expr': label_to_expr,
                }, checkpoint_path)

                # Manage checkpoint rotation
                manage_checkpoints(FLAGS.checkpoint_dir, FLAGS.max_checkpoints_to_keep)

                # Evaluate on test set
                print("Evaluating on test set...")
                test_loss = evaluate_model(
                    model, test_loader, device, vocab_size,
                    label_to_expr, end_token, FLAGS.output_dir, global_step
                )
                print(f"Test loss: {test_loss:.4f}")

                model.train()  # Return to training mode

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

    print("Training completed!")

    # Final evaluation
    print("Running final evaluation...")
    test_loss = evaluate_model(
        model, test_loader, device, vocab_size,
        label_to_expr, end_token, FLAGS.output_dir, 'final'
    )
    print(f"Final test loss: {test_loss:.4f}")

    # Save final model
    final_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'end_token': end_token,
        'label_to_expr': label_to_expr,
        'config': {
            'image_size': FLAGS.image_size,
            'ngram_length': FLAGS.ngram_length,
            'hidden_dim': FLAGS.hidden_dim,
            'also_train_vgg': FLAGS.also_train_vgg,
        }
    }, final_checkpoint_path)


if __name__ == '__main__':
    app.run(main)