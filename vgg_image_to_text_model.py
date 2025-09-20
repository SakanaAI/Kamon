"""VGG-based image-to-text model for Japanese Kamon description generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from typing import Tuple, List, Optional


class VGGImageToTextModel(nn.Module):
    """VGG-based n-gram language model for image-to-text generation.

    This model uses trainable masks at each position to selectively mask parts
    of the input image, then feeds masked images through a shared VGG16 feature
    extractor. Features are combined with previous position features and logits
    to predict the next token.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        image_size: int = 224,
        ngram_length: int = 2,
        vgg_feature_dim: int = 4096,
        hidden_dim: int = 512,
        also_train_vgg: bool = False,
    ):
        """Initialize the model.

        Args:
            vocab_size: Size of the vocabulary
            max_seq_len: Maximum sequence length (including EOS token)
            image_size: Input image size (assumes square images)
            ngram_length: N-gram context length (2 for bigram, 3 for trigram, etc.)
            vgg_feature_dim: Dimension of VGG features (4096 for VGG16 classifier)
            hidden_dim: Hidden dimension for combining features
            also_train_vgg: Whether to train VGG parameters or freeze them
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.ngram_length = ngram_length
        self.vgg_feature_dim = vgg_feature_dim
        self.hidden_dim = hidden_dim
        self.also_train_vgg = also_train_vgg

        # Initialize VGG16 feature extractor
        self.construct_vgg_classifier()

        # Trainable masks for each position - sigmoid to keep values in [0,1]
        # Shape: (max_seq_len, 1, image_size, image_size)
        self.position_masks = nn.Parameter(
            torch.zeros(max_seq_len, 1, image_size, image_size)
        )

        # Linear layers for combining features
        # Input: current VGG features + (ngram_length - 1) * (previous VGG features + previous logits)
        # = vgg_feature_dim + (ngram_length - 1) * (vgg_feature_dim + vocab_size)
        input_dim = vgg_feature_dim + (ngram_length - 1) * (vgg_feature_dim + vocab_size)
        self.feature_combiner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Output classifier
        self.classifier = nn.Linear(hidden_dim, vocab_size)

        # Dummy features and logits for initial position
        self.register_buffer('dummy_features', torch.zeros(1, vgg_feature_dim))
        self.register_buffer('dummy_logits', torch.zeros(1, vocab_size))

    def construct_vgg_classifier(self):
        """Initialize VGG16 model with classifier features exposed."""
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Remove the last classification layer to expose features
        vgg_model.classifier = vgg_model.classifier[:-1]

        params = list(vgg_model.parameters())
        if not self.also_train_vgg:
            for p in params:
                p.requires_grad = False
            vgg_model.eval()

        # Get feature dimension from last layer
        self.vgg_nfeatures = params[-1].shape[0] if params else 4096
        self.vgg = vgg_model

    def apply_mask_to_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to image.

        Args:
            image: Input image tensor [B, 3, H, W]
            mask: Mask tensor [1, H, W] or [B, 1, H, W]

        Returns:
            Masked image tensor [B, 3, H, W]
        """
        # Apply sigmoid to mask to ensure values are in [0, 1]
        mask = torch.sigmoid(mask)

        # Expand mask to match image dimensions if needed
        if mask.dim() == 3:  # [1, H, W]
            mask = mask.unsqueeze(0)  # [1, 1, H, W]
        if mask.size(0) == 1 and image.size(0) > 1:
            mask = mask.expand(image.size(0), -1, -1, -1)  # [B, 1, H, W]

        # Expand mask to cover all color channels
        mask = mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

        return image * mask

    def extract_vgg_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using VGG16.

        Args:
            image: Input image tensor [B, 3, H, W]

        Returns:
            Feature tensor [B, vgg_feature_dim]
        """
        if not self.also_train_vgg:
            with torch.no_grad():
                return self.vgg(image)
        else:
            return self.vgg(image)

    def forward(
        self,
        images: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            images: Input images [B, 3, H, W]
            target_tokens: Target token sequences [B, max_seq_len] (for training)

        Returns:
            Tuple of (all_logits, all_masks):
                all_logits: Predicted logits for each position [B, max_seq_len, vocab_size]
                all_masks: Applied masks for each position [B, max_seq_len, H, W]
        """
        batch_size = images.size(0)
        device = images.device

        all_logits = []
        all_masks = []

        # Initialize history buffers for n-gram context
        # Store the last (ngram_length - 1) features and logits
        feature_history = []
        logits_history = []

        for pos in range(self.max_seq_len):
            # Get mask for current position
            current_mask = self.position_masks[pos]  # [1, H, W]

            # Apply mask to images
            masked_images = self.apply_mask_to_image(images, current_mask)

            # Extract VGG features from masked images
            vgg_features = self.extract_vgg_features(masked_images)

            # Build context from history (n-gram context)
            # We need exactly (ngram_length - 1) previous contexts
            context_features = []
            context_logits = []

            for i in range(self.ngram_length - 1):
                if i < len(feature_history):
                    # Use actual history (most recent first)
                    hist_idx = len(feature_history) - 1 - i
                    context_features.append(feature_history[hist_idx])
                    context_logits.append(logits_history[hist_idx])
                else:
                    # Use dummy features/logits for missing history
                    context_features.append(self.dummy_features.expand(batch_size, -1))
                    context_logits.append(self.dummy_logits.expand(batch_size, -1))

            # Combine: current VGG features + interleaved (previous features, previous logits)
            combined_parts = [vgg_features]
            for feat, logit in zip(context_features, context_logits):
                combined_parts.extend([feat, logit])

            combined_input = torch.cat(combined_parts, dim=1)

            # Pass through feature combiner
            hidden_features = self.feature_combiner(combined_input)

            # Generate logits for current position
            logits = self.classifier(hidden_features)

            all_logits.append(logits)
            all_masks.append(torch.sigmoid(current_mask).expand(batch_size, -1, -1))  # Expand to batch size

            # Update history buffers
            feature_history.append(vgg_features)
            logits_history.append(logits)

            # Keep only the last (ngram_length - 1) entries to maintain n-gram context
            if len(feature_history) > self.ngram_length - 1:
                feature_history.pop(0)
                logits_history.pop(0)

        # Stack results
        all_logits = torch.stack(all_logits, dim=1)  # [B, max_seq_len, vocab_size]
        all_masks = torch.stack(all_masks, dim=1)    # [B, max_seq_len, H, W]

        return all_logits, all_masks

    def generate(
        self,
        images: torch.Tensor,
        end_token: int,
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate text descriptions for images.

        Args:
            images: Input images [B, 3, H, W]
            end_token: End token ID
            max_length: Maximum generation length (defaults to max_seq_len)

        Returns:
            Tuple of (generated_tokens, all_masks):
                generated_tokens: Generated token sequences [B, seq_len]
                all_masks: Applied masks for each position [B, seq_len, H, W]
        """
        if max_length is None:
            max_length = self.max_seq_len

        batch_size = images.size(0)
        device = images.device

        generated_tokens = []
        all_masks = []

        # Initialize history buffers for n-gram context
        feature_history = []
        logits_history = []

        for pos in range(max_length):
            # Get mask for current position
            current_mask = self.position_masks[min(pos, self.max_seq_len - 1)]

            # Apply mask to images
            masked_images = self.apply_mask_to_image(images, current_mask)

            # Extract VGG features from masked images
            vgg_features = self.extract_vgg_features(masked_images)

            # Build context from history (n-gram context)
            # We need exactly (ngram_length - 1) previous contexts
            context_features = []
            context_logits = []

            for i in range(self.ngram_length - 1):
                if i < len(feature_history):
                    # Use actual history (most recent first)
                    hist_idx = len(feature_history) - 1 - i
                    context_features.append(feature_history[hist_idx])
                    context_logits.append(logits_history[hist_idx])
                else:
                    # Use dummy features/logits for missing history
                    context_features.append(self.dummy_features.expand(batch_size, -1))
                    context_logits.append(self.dummy_logits.expand(batch_size, -1))

            # Combine: current VGG features + interleaved (previous features, previous logits)
            combined_parts = [vgg_features]
            for feat, logit in zip(context_features, context_logits):
                combined_parts.extend([feat, logit])

            combined_input = torch.cat(combined_parts, dim=1)

            # Pass through feature combiner
            hidden_features = self.feature_combiner(combined_input)

            # Generate logits for current position
            logits = self.classifier(hidden_features)

            # Sample tokens (greedy decoding)
            tokens = torch.argmax(logits, dim=1)

            generated_tokens.append(tokens)
            all_masks.append(torch.sigmoid(current_mask).expand(batch_size, -1, -1))

            # Update history buffers
            feature_history.append(vgg_features)
            logits_history.append(logits)

            # Keep only the last (ngram_length - 1) entries to maintain n-gram context
            if len(feature_history) > self.ngram_length - 1:
                feature_history.pop(0)
                logits_history.pop(0)

            # Check if all sequences have generated end token
            if torch.all(tokens == end_token):
                break

        # Stack results
        generated_tokens = torch.stack(generated_tokens, dim=1)  # [B, seq_len]
        all_masks = torch.stack(all_masks, dim=1)                # [B, seq_len, H, W]

        return generated_tokens, all_masks