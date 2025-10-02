#!/usr/bin/env python3
"""Standalone inference script for VGG-based Kamon image-to-text model."""

import os
import sys
import json
import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from absl import app, flags
import numpy as np
import jaconv

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

import kamon_dataset as kd
from vgg_image_to_text_model import VGGImageToTextModel

# Define command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_path', '', 'Path to checkpoint file (.pt)')
flags.DEFINE_string('dataset_subset', 'test', 'Dataset subset: train, val, or test')
flags.DEFINE_boolean('omit_edo', True, 'Whether to omit Edo period images')
flags.DEFINE_string('output_file', 'inference_results.jsonl', 'Output JSONL file')
flags.DEFINE_integer('batch_size', 16, 'Batch size for inference')
flags.DEFINE_string('device', 'auto', 'Device to use (cuda, cpu, or auto)')


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and return model and metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration from checkpoint
    vocab_size = checkpoint['vocab_size']
    max_seq_len = checkpoint['max_seq_len']
    end_token = checkpoint['end_token']
    label_to_expr = checkpoint['label_to_expr']

    # Try to get model config, with fallbacks for older checkpoints
    model_config = checkpoint.get('config', {})
    image_size = model_config.get('image_size', 224)
    hidden_dim = model_config.get('hidden_dim', 512)
    also_train_vgg = model_config.get('also_train_vgg', False)
    use_masks = model_config.get('use_masks', True)  # Default to True for backward compatibility

    # Infer ngram_length from the feature_combiner input dimension if not in config
    if 'ngram_length' in model_config:
        ngram_length = model_config['ngram_length']
    else:
        # Infer from feature_combiner.0.weight shape
        # Input dim = vgg_feature_dim + (ngram_length - 1) * (vgg_feature_dim + vocab_size)
        # Assuming vgg_feature_dim = 4096
        vgg_feature_dim = 4096
        feature_combiner_input_dim = checkpoint['model_state_dict']['feature_combiner.0.weight'].shape[1]

        # Solve: feature_combiner_input_dim = vgg_feature_dim + (ngram_length - 1) * (vgg_feature_dim + vocab_size)
        # Rearrange: (feature_combiner_input_dim - vgg_feature_dim) = (ngram_length - 1) * (vgg_feature_dim + vocab_size)
        # ngram_length = 1 + (feature_combiner_input_dim - vgg_feature_dim) / (vgg_feature_dim + vocab_size)
        ngram_length = 1 + (feature_combiner_input_dim - vgg_feature_dim) // (vgg_feature_dim + vocab_size)
        print(f"Inferred ngram_length = {ngram_length} from checkpoint dimensions")

    # Create model
    model = VGGImageToTextModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        image_size=image_size,
        ngram_length=ngram_length,
        hidden_dim=hidden_dim,
        also_train_vgg=also_train_vgg,
        use_masks=use_masks,
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    metadata = {
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'end_token': end_token,
        'label_to_expr': label_to_expr,
        'image_size': image_size,
        'step': checkpoint.get('step', 'unknown'),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', checkpoint.get('loss', 'unknown'))
    }

    return model, metadata


def normalize_description(desc):
    """Normalize description by removing whitespace and converting to hiragana.

    Args:
        desc: Description string

    Returns:
        Normalized description with spaces removed and katakana converted to hiragana
    """
    # Remove all whitespace
    desc = desc.replace(' ', '').replace('\t', '').replace('\n', '')
    # Convert katakana to hiragana for consistent comparison
    desc = jaconv.kata2hira(desc)
    return desc


def build_description_to_images_map(train_metadata):
    """Build a mapping from descriptions to lists of image paths in training data.

    Args:
        train_metadata: List of metadata dicts from training dataset

    Returns:
        Dictionary mapping normalized description strings to lists of image paths
    """
    desc_to_images = {}
    for item in train_metadata:
        desc = item.get('description', '')
        img_path = item.get('path', '')
        if desc and img_path:
            # Normalize description by removing spaces for consistent lookup
            normalized_desc = normalize_description(desc)
            if normalized_desc not in desc_to_images:
                desc_to_images[normalized_desc] = []
            desc_to_images[normalized_desc].append(img_path)
    return desc_to_images


def run_inference(model, dataloader, device, label_to_expr, end_token, dataset_metadata, train_desc_to_images):
    """Run inference on dataset and return results."""
    model.eval()
    results = []

    print(f"Running inference on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, (images, target_tokens) in enumerate(dataloader):
            images = images.to(device)
            target_tokens = target_tokens.to(device)

            # Generate predictions
            pred_tokens, pred_masks = model.generate(images, end_token)

            batch_size = images.size(0)

            # Process each example in the batch
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

                # Get image path and other metadata from the dataset
                example_idx = batch_idx * dataloader.batch_size + i
                if example_idx < len(dataset_metadata):
                    item_metadata = dataset_metadata[example_idx]
                    image_path = item_metadata.get('path', '')
                    translation = item_metadata.get('translation', '')
                else:
                    image_path = ''
                    translation = ''

                # Lookup training images with same description as reference
                # Normalize descriptions by removing spaces before lookup
                normalized_reference = normalize_description(reference_description)
                train_images_reference = train_desc_to_images.get(normalized_reference, [])

                # Lookup training images with same description as predicted
                normalized_predicted = normalize_description(predicted_description)
                train_images_predicted = train_desc_to_images.get(normalized_predicted, [])

                result = {
                    'reference': reference_description,
                    'predicted': predicted_description,
                    'image': image_path,
                    'translation': translation,
                    'train_images_reference': train_images_reference,
                    'train_images_predicted': train_images_predicted,
                    'reference_tokens': gt_tokens_list,
                    'predicted_tokens': pred_tokens_list,
                    'batch_idx': batch_idx,
                    'example_idx': i
                }

                results.append(result)

            # Progress reporting
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    return results


def main(argv):
    del argv  # Unused

    # Validate required arguments
    if not FLAGS.checkpoint_path:
        print("Error: --checkpoint_path is required")
        return 1

    if FLAGS.dataset_subset not in ['train', 'val', 'test']:
        print("Error: --dataset_subset must be one of: train, val, test")
        return 1

    # Set device
    if FLAGS.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(FLAGS.device)

    print(f"Using device: {device}")

    try:
        # Load checkpoint and create model
        model, checkpoint_metadata = load_checkpoint(FLAGS.checkpoint_path, device)

        print(f"Loaded model from step {checkpoint_metadata['step']}, epoch {checkpoint_metadata['epoch']}")
        print(f"Vocabulary size: {checkpoint_metadata['vocab_size']}")
        print(f"Max sequence length: {checkpoint_metadata['max_seq_len']}")
        if checkpoint_metadata['val_loss'] != 'unknown':
            print(f"Validation loss: {checkpoint_metadata['val_loss']}")

        # Load training dataset to build description-to-images mapping
        print(f"Loading training dataset for description lookup...")
        train_dataset = kd.KamonDataset(
            division="train",
            image_size=checkpoint_metadata['image_size'],
            num_augmentations=0,  # No augmentation needed for lookup
            one_hot=False,
            omit_edo=FLAGS.omit_edo,
        )
        train_desc_to_images = build_description_to_images_map(train_dataset.metadata)
        print(f"Built training description map with {len(train_desc_to_images)} unique descriptions")

        # Load evaluation dataset
        print(f"Loading {FLAGS.dataset_subset} dataset (omit_edo={FLAGS.omit_edo})...")
        dataset = kd.KamonDataset(
            division=FLAGS.dataset_subset,
            image_size=checkpoint_metadata['image_size'],
            num_augmentations=0,  # No augmentation for inference
            one_hot=False,
            omit_edo=FLAGS.omit_edo,
        )

        print(f"Dataset size: {len(dataset)} examples")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=False,  # Keep original order for inference
            num_workers=4,
            pin_memory=True
        )

        # Get dataset metadata for image paths and translations
        dataset_metadata = dataset.metadata

        # Run inference
        results = run_inference(
            model,
            dataloader,
            device,
            checkpoint_metadata['label_to_expr'],
            checkpoint_metadata['end_token'],
            dataset_metadata,
            train_desc_to_images
        )

        # Save results
        print(f"Saving results to: {FLAGS.output_file}")
        with jsonlines.open(FLAGS.output_file, 'w') as writer:
            for result in results:
                writer.write(result)

        # Calculate and print summary statistics
        total_examples = len(results)
        correct_predictions = sum(1 for r in results if r['reference'] == r['predicted'])
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0

        print(f"\nInference completed!")
        print(f"Total examples: {total_examples}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Results saved to: {FLAGS.output_file}")

        return 0

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    app.run(main)