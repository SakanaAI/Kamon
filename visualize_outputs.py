#!/usr/bin/env python3
"""Generate HTML visualization for Kamon model outputs."""

import json
import jsonlines
import html
import os
import base64
import argparse
from typing import List, Tuple, Dict, Any


def calculate_character_error_rate(reference: str, predicted: str) -> Tuple[int, int, int, float]:
    """Calculate character-level edit distance and error rate.

    Args:
        reference: Ground truth string
        predicted: Predicted string

    Returns:
        Tuple of (insertions, deletions, substitutions, error_rate)
    """
    ref_chars = list(reference)
    pred_chars = list(predicted)

    # Dynamic programming table for edit distance
    dp = [[0] * (len(pred_chars) + 1) for _ in range(len(ref_chars) + 1)]

    # Initialize first row and column
    for i in range(len(ref_chars) + 1):
        dp[i][0] = i  # deletions
    for j in range(len(pred_chars) + 1):
        dp[0][j] = j  # insertions

    # Fill the DP table
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(pred_chars) + 1):
            if ref_chars[i-1] == pred_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]  # no operation
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )

    # Backtrack to count operations
    i, j = len(ref_chars), len(pred_chars)
    insertions = deletions = substitutions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i-1] == pred_chars[j-1]:
            # Match - no operation
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion
            insertions += 1
            j -= 1

    total_operations = insertions + deletions + substitutions
    error_rate = total_operations / max(len(ref_chars), 1) if len(ref_chars) > 0 else float('inf')

    return insertions, deletions, substitutions, error_rate


def create_aligned_display(reference: str, predicted: str, translation: str = "") -> str:
    """Create HTML for aligned display of reference and predicted strings."""
    ref_html = html.escape(reference)
    pred_html = html.escape(predicted)
    trans_html = html.escape(translation) if translation else ""

    translation_line = f"""
        <div class="translation-line">
            <span class="label">EN:</span>
            <span class="text translation">{trans_html}</span>
        </div>
    """ if translation else ""

    return f"""
    <div class="alignment-container">
        <div class="reference-line">
            <span class="label">REF:</span>
            <span class="text">{ref_html}</span>
        </div>
        <div class="predicted-line">
            <span class="label">PRD:</span>
            <span class="text">{pred_html}</span>
        </div>
        {translation_line}
    </div>
    """


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for embedding in HTML."""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            # Determine image format from extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')
            return f"data:{mime_type};base64,{img_base64}"
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return ""


def generate_html_page(jsonl_file: str, output_file: str) -> None:
    """Generate HTML page from JSONL file."""

    # Load data
    data = []
    with jsonlines.open(jsonl_file) as reader:
        for item in reader:
            data.append(item)

    if not data:
        print("No data found in the JSONL file.")
        return

    # Calculate overall character error rate
    total_insertions = total_deletions = total_substitutions = 0
    total_ref_chars = 0

    for item in data:
        ref = item.get('reference', '')
        pred = item.get('predicted', '')
        ins, dels, subs, _ = calculate_character_error_rate(ref, pred)
        total_insertions += ins
        total_deletions += dels
        total_substitutions += subs
        total_ref_chars += len(ref)

    total_operations = total_insertions + total_deletions + total_substitutions
    overall_cer = total_operations / max(total_ref_chars, 1) if total_ref_chars > 0 else 0.0

    # Find cases where prediction = reference AND no training images
    perfect_predictions_no_training = []
    for idx, item in enumerate(data):
        reference = item.get('reference', '')
        predicted = item.get('predicted', '')
        train_images_reference = item.get('train_images_reference', [])
        train_images_predicted = item.get('train_images_predicted', [])

        if (reference == predicted and
            len(train_images_reference) == 0 and
            len(train_images_predicted) == 0):
            perfect_predictions_no_training.append((idx + 1, reference))

    # Generate navigation links
    navigation_html = ""
    if perfect_predictions_no_training:
        navigation_html = f"""
        <div class="navigation-section">
            <h3>Perfect Predictions with No Training Support ({len(perfect_predictions_no_training)} cases)</h3>
            <p>These are cases where the model correctly predicted the reference description, but there were no training images with the same description:</p>
            <div class="navigation-links">
                {''.join([f'<a href="#example-{idx}" class="nav-link">Example {idx}: {html.escape(desc)}</a>' for idx, desc in perfect_predictions_no_training])}
            </div>
        </div>
        """

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kamon Model Output Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: -20px -20px 30px -20px;
            border-radius: 0 0 15px 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin: 0;
            font-size: 2em;
            text-align: center;
        }}

        .cer-stats {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            margin-top: 15px;
            border-radius: 10px;
            text-align: center;
        }}

        .cer-stats h2 {{
            margin: 0 0 10px 0;
            font-size: 1.5em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}

        .stat-item {{
            background: rgba(255,255,255,0.1);
            padding: 10px;
            border-radius: 8px;
        }}

        .stat-value {{
            font-size: 1.3em;
            font-weight: bold;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .item {{
            background: white;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}

        .item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}

        .image-container {{
            text-align: center;
            margin-bottom: 15px;
        }}

        .main-image {{
            max-width: 300px;
            max-height: 300px;
            border: 2px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .alignment-container {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }}

        .reference-line, .predicted-line, .translation-line {{
            margin: 5px 0;
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }}

        .reference-line .label {{
            display: inline-block;
            width: 40px;
            font-weight: bold;
            color: #28a745;
        }}

        .predicted-line .label {{
            display: inline-block;
            width: 40px;
            font-weight: bold;
            color: #dc3545;
        }}

        .translation-line .label {{
            display: inline-block;
            width: 40px;
            font-weight: bold;
            color: #6c757d;
        }}

        .translation {{
            font-style: italic;
            color: #495057;
        }}

        .text {{
            background: white;
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}

        .training-images {{
            margin-top: 20px;
        }}

        .training-images h4 {{
            margin-bottom: 10px;
            color: #666;
            font-size: 0.9em;
        }}

        .training-images-container {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .training-image {{
            max-width: 80px;
            max-height: 80px;
            border: 1px solid #ddd;
            border-radius: 6px;
            opacity: 0.8;
            transition: opacity 0.2s ease;
        }}

        .training-image:hover {{
            opacity: 1;
        }}

        .item-number {{
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 15px;
        }}

        .error-info {{
            font-size: 0.8em;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }}

        .navigation-section {{
            background: white;
            margin: 20px -20px 30px -20px;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .navigation-section h3 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .navigation-links {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}

        .nav-link {{
            display: block;
            padding: 8px 12px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            text-decoration: none;
            color: #495057;
            font-size: 0.9em;
            transition: all 0.2s ease;
        }}

        .nav-link:hover {{
            background: #e9ecef;
            border-color: #667eea;
            color: #667eea;
            transform: translateY(-1px);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Kamon Model Output Visualization</h1>
        <div class="cer-stats">
            <h2>Overall Character Error Rate: {overall_cer:.3f} ({overall_cer*100:.1f}%)</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_insertions}</div>
                    <div>Insertions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_deletions}</div>
                    <div>Deletions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_substitutions}</div>
                    <div>Substitutions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_ref_chars}</div>
                    <div>Total Ref Chars</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(data)}</div>
                    <div>Total Examples</div>
                </div>
            </div>
        </div>
    </div>

    {navigation_html}

    <div class="container">
"""

    # Add each item
    for idx, item in enumerate(data):
        reference = item.get('reference', '')
        predicted = item.get('predicted', '')
        image_path = item.get('image', '')
        train_images_reference = item.get('train_images_reference', [])
        train_images_predicted = item.get('train_images_predicted', [])
        translation = item.get('translation', '')

        # Convert main image to base64
        main_image_data = image_to_base64(image_path) if image_path else ""

        # Create alignment display
        alignment_html = create_aligned_display(reference, predicted, translation)

        # Training images handling
        training_images_html = ""

        # Check if reference and predicted are identical
        if reference == predicted:
            # When identical, combine the images and show with unified label
            all_training_images = list(set(train_images_reference + train_images_predicted))  # Remove duplicates
            if all_training_images:
                img_tags = []
                for train_img_path in all_training_images:
                    train_img_data = image_to_base64(train_img_path)
                    if train_img_data:
                        img_tags.append(f'<img src="{train_img_data}" class="training-image" alt="Training image matching both reference and prediction">')

                if img_tags:
                    training_images_html = f"""
                    <div class="training-images">
                        <h4>Training images with same description as prediction and reference:</h4>
                        <div class="training-images-container">
                            {''.join(img_tags)}
                        </div>
                    </div>
                    """
        else:
            # When different, show separate sections
            if train_images_reference:
                ref_img_tags = []
                for train_img_path in train_images_reference:
                    train_img_data = image_to_base64(train_img_path)
                    if train_img_data:
                        ref_img_tags.append(f'<img src="{train_img_data}" class="training-image" alt="Training image matching reference">')

                if ref_img_tags:
                    training_images_html += f"""
                    <div class="training-images">
                        <h4>Training images with same description as reference:</h4>
                        <div class="training-images-container">
                            {''.join(ref_img_tags)}
                        </div>
                    </div>
                    """

            if train_images_predicted:
                pred_img_tags = []
                for train_img_path in train_images_predicted:
                    train_img_data = image_to_base64(train_img_path)
                    if train_img_data:
                        pred_img_tags.append(f'<img src="{train_img_data}" class="training-image" alt="Training image matching prediction">')

                if pred_img_tags:
                    training_images_html += f"""
                    <div class="training-images">
                        <h4>Training images with same description as prediction:</h4>
                        <div class="training-images-container">
                            {''.join(pred_img_tags)}
                        </div>
                    </div>
                    """

        # Calculate individual error rates
        ins, dels, subs, cer = calculate_character_error_rate(reference, predicted)
        error_info = f"Individual CER: {cer:.3f} (I:{ins}, D:{dels}, S:{subs})"

        item_html = f"""
        <div class="item" id="example-{idx + 1}">
            <div class="item-number">Example {idx + 1}</div>

            <div class="image-container">
                {"<img src='" + main_image_data + "' class='main-image' alt='Kamon image'>" if main_image_data else "<p>Image not found: " + html.escape(image_path) + "</p>"}
            </div>

            {alignment_html}

            <div class="error-info">{error_info}</div>

            {training_images_html}
        </div>
        """

        html_content += item_html

    html_content += """
    </div>
</body>
</html>
"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML visualization generated: {output_file}")
    print(f"Overall Character Error Rate: {overall_cer:.3f} ({overall_cer*100:.1f}%)")
    print(f"Total examples: {len(data)}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML visualization for Kamon model outputs')
    parser.add_argument('input_jsonl', help='Input JSONL file with model outputs')
    parser.add_argument('-o', '--output', default='visualization.html',
                       help='Output HTML file (default: visualization.html)')

    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        print(f"Error: Input file {args.input_jsonl} does not exist")
        return

    generate_html_page(args.input_jsonl, args.output)


if __name__ == '__main__':
    main()