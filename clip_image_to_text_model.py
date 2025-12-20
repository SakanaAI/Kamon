"""CLIP-based image-to-text model for Japanese Kamon description generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Tuple, List, Optional

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft library not available. LoRA support disabled.")


class CLIPImageToTextModel(nn.Module):
    """CLIP-based n-gram language model for image-to-text generation.

    This model uses trainable masks at each position to selectively mask parts
    of the input image, then feeds masked images through a shared CLIP vision
    encoder. Features are combined with previous position features and logits
    to predict the next token.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        image_size: int = 224,
        ngram_length: int = 2,
        hidden_dim: int = 512,
        also_train_clip: bool = False,
        use_masks: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        """Initialize the model.

        Args:
            vocab_size: Size of the vocabulary
            max_seq_len: Maximum sequence length (including EOS token)
            image_size: Input image size (assumes square images)
            ngram_length: N-gram context length (2 for bigram, 3 for trigram, etc.)
            hidden_dim: Hidden dimension for combining features
            also_train_clip: Whether to train CLIP parameters or freeze them
            use_masks: Whether to use position-specific trainable masks
            clip_model_name: Name of the CLIP model to use
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA rank (low-rank dimension)
            lora_alpha: LoRA alpha (scaling parameter)
            lora_dropout: LoRA dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.ngram_length = ngram_length
        self.hidden_dim = hidden_dim
        self.also_train_clip = also_train_clip
        self.use_masks = use_masks
        self.clip_model_name = clip_model_name
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Initialize CLIP vision encoder
        self.construct_clip_encoder()

        # Trainable masks for each position - sigmoid to keep values in [0,1]
        # Shape: (max_seq_len, 1, image_size, image_size)
        # Initialize with small random values to break symmetry and encourage learning
        if self.use_masks:
            self.position_masks = nn.Parameter(
                torch.randn(max_seq_len, 1, image_size, image_size) * 0.1
            )
        else:
            # Create non-trainable dummy masks (all ones) when masks are disabled
            self.register_buffer(
                'position_masks',
                torch.ones(max_seq_len, 1, image_size, image_size)
            )

        # Linear layers for combining features
        # Input: current CLIP features + (ngram_length - 1) * (previous CLIP features + previous logits)
        # = clip_feature_dim + (ngram_length - 1) * (clip_feature_dim + vocab_size)
        input_dim = self.clip_feature_dim + (ngram_length - 1) * (self.clip_feature_dim + vocab_size)
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
        self.register_buffer('dummy_features', torch.zeros(1, self.clip_feature_dim))
        self.register_buffer('dummy_logits', torch.zeros(1, vocab_size))

    def construct_clip_encoder(self):
        """Initialize CLIP vision encoder with optional LoRA."""
        self.clip_vision = CLIPVisionModel.from_pretrained(self.clip_model_name)

        # Get feature dimension from CLIP model config
        self.clip_feature_dim = self.clip_vision.config.hidden_size

        # Apply LoRA if requested
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft library is required for LoRA. Install it with: pip install peft")

            print(f"Applying LoRA to CLIP vision encoder (r={self.lora_r}, alpha={self.lora_alpha})")

            # Configure LoRA to target attention layers in the vision transformer
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Attention layers
                lora_dropout=self.lora_dropout,
                bias="none",
                modules_to_save=[],  # Don't save any non-LoRA modules
            )

            # Apply LoRA - this freezes base model and adds trainable LoRA adapters
            self.clip_vision = get_peft_model(self.clip_vision, lora_config)
            self.clip_vision.print_trainable_parameters()
        elif not self.also_train_clip:
            # Freeze CLIP if not using LoRA and not training full model
            for p in self.clip_vision.parameters():
                p.requires_grad = False
            self.clip_vision.eval()

        # Initialize image processor for CLIP
        # Note: We'll apply our own normalization to work with masked images
        self.clip_processor = CLIPImageProcessor.from_pretrained(self.clip_model_name)

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

    def extract_clip_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using CLIP vision encoder.

        Args:
            image: Input image tensor [B, 3, H, W]

        Returns:
            Feature tensor [B, clip_feature_dim]
        """
        # If using LoRA or training full CLIP, compute gradients
        if self.use_lora or self.also_train_clip:
            outputs = self.clip_vision(pixel_values=image)
            return outputs.pooler_output
        else:
            # Frozen CLIP without LoRA
            with torch.no_grad():
                outputs = self.clip_vision(pixel_values=image)
                return outputs.pooler_output

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

            # Extract CLIP features from masked images
            clip_features = self.extract_clip_features(masked_images)

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

            # Combine: current CLIP features + interleaved (previous features, previous logits)
            combined_parts = [clip_features]
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
            feature_history.append(clip_features)
            logits_history.append(logits)

            # Keep only the last (ngram_length - 1) entries to maintain n-gram context
            if len(feature_history) > self.ngram_length - 1:
                feature_history.pop(0)
                logits_history.pop(0)

        # Stack results
        all_logits = torch.stack(all_logits, dim=1)  # [B, max_seq_len, vocab_size]
        all_masks = torch.stack(all_masks, dim=1)    # [B, max_seq_len, H, W]

        return all_logits, all_masks

    def get_mask_diversity_loss(self, weight=0.01):
        """Calculate a regularization loss to encourage mask diversity between positions.

        Args:
            weight: Weight for the diversity loss term

        Returns:
            Diversity loss encouraging different masks at different positions (0 if masks disabled)
        """
        # Return zero loss if masks are not being used
        if not self.use_masks:
            return torch.tensor(0.0, device=self.position_masks.device)

        # Get masks after sigmoid
        masks = torch.sigmoid(self.position_masks)  # [seq_len, 1, H, W]

        # Flatten spatial dimensions for easier computation
        masks_flat = masks.view(self.max_seq_len, -1)  # [seq_len, H*W]

        # Compute pairwise similarities between different positions
        similarities = torch.mm(masks_flat, masks_flat.t())  # [seq_len, seq_len]

        # Remove diagonal (self-similarities) and take mean of off-diagonal elements
        mask_diag = torch.eye(self.max_seq_len, device=masks.device, dtype=masks.dtype)
        off_diagonal = similarities * (1 - mask_diag)

        # Higher similarity = higher loss (we want different masks)
        diversity_loss = off_diagonal.sum() / (self.max_seq_len * (self.max_seq_len - 1))

        return weight * diversity_loss

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

            # Extract CLIP features from masked images
            clip_features = self.extract_clip_features(masked_images)

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

            # Combine: current CLIP features + interleaved (previous features, previous logits)
            combined_parts = [clip_features]
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
            feature_history.append(clip_features)
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
