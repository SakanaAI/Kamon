import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

import seaborn as sns

from matplotlib import patheffects
from scipy import ndimage


mpl.use('Agg')

from tqdm.auto import tqdm

def find_center_of_mass(array_2d):
    """
    Alternative implementation using np.average and meshgrid.
    This version is generally faster and more concise.

    Args:
        array_2d: A 2D numpy array of values between 0 and 1.

    Returns:
        A tuple (x, y) representing the coordinates of the center of mass.
    """
    total_mass = np.sum(array_2d)
    if total_mass == 0:
      return (np.nan, np.nan)

    y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
    x_center = np.average(x_coords, weights=array_2d)
    y_center = np.average(y_coords, weights=array_2d)
    return (round(y_center, 4), round(x_center, 4))

def draw_path(x, route, valid_only=False, gt=False, cmap=None):
    """
    Draws a path on a maze image based on a given route.

    Args:
        maze: A numpy array representing the maze image.
        route: A list of integers representing the route, where 0 is up, 1 is down, 2 is left, and 3 is right.
        valid_only: A boolean indicating whether to only draw valid steps (i.e., steps that don't go into walls).

    Returns:
        A numpy array representing the maze image with the path drawn in blue.
    """
    x = np.copy(x)
    start = np.argwhere((x == [1, 0, 0]).all(axis=2))
    end = np.argwhere((x == [0, 1, 0]).all(axis=2))
    if cmap is None:
        cmap = plt.get_cmap('winter') if not valid_only else  plt.get_cmap('summer')

    # Initialize the current position
    current_pos = start[0]

    # Draw the path
    colors = cmap(np.linspace(0, 1, len(route)))
    si = 0
    for step in route:
        new_pos = current_pos
        if step == 0:  # Up
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif step == 1:  # Down
            new_pos = (current_pos[0] + 1, current_pos[1])
        elif step == 2:  # Left
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif step == 3:  # Right
            new_pos = (current_pos[0], current_pos[1] + 1)
        elif step == 4:  # Do nothing
            pass
        else:
            raise ValueError("Invalid step: {}".format(step))

        # Check if the new position is valid
        if valid_only:
            try:
                if np.all(x[new_pos] == [0,0,0]):  # Check if it's a wall
                    continue  # Skip this step if it's invalid
            except IndexError:
                continue  # Skip this step if it's out of bounds

        # Draw the step
        if new_pos[0] >= 0 and new_pos[0] < x.shape[0] and new_pos[1] >= 0 and new_pos[1] < x.shape[1]:
            if not ((x[new_pos] == [1,0,0]).all() or (x[new_pos] == [0,1,0]).all()):
                colour = colors[si][:3]
                si += 1
                x[new_pos] = x[new_pos]*0.5 + colour*0.5

        # Update the current position
        current_pos = new_pos
        # cv2.imwrite('maze2.png', x[:,:,::-1]*255)

    return x


def find_island_centers(array_2d, threshold):
    """
    Finds the center of mass of each island (connected component) in a 2D array.

    Args:
        array_2d: A 2D numpy array of values.
        threshold: The threshold to binarize the array.

    Returns:
        A list of tuples (y, x) representing the center of mass of each island.
    """
    binary_image = array_2d > threshold
    labeled_image, num_labels = ndimage.label(binary_image)
    centers = []
    areas = []  # Store the area of each island
    for i in range(1, num_labels + 1):
        island = (labeled_image == i)
        total_mass = np.sum(array_2d[island])
        if total_mass > 0:
            y_coords, x_coords = np.mgrid[:array_2d.shape[0], :array_2d.shape[1]]
            x_center = np.average(x_coords[island], weights=array_2d[island])
            y_center = np.average(y_coords[island], weights=array_2d[island])
            centers.append((round(y_center, 4), round(x_center, 4)))
            areas.append(np.sum(island))  # Calculate area of the island
    return centers, areas


# And add this to the plotting utility for your task (MAKE SURE TO IMPORT IT):
def make_kamon_gif(image, targets, predictions, certainties, attention_tracking,
                   class_labels, save_location, bi, index):
    # Shapes:
    # image: [C, H, W]
    # targets: [Description length]
    # predictions: [length, classes, ticks]
    # certainties: [2, ticks]
    # attention_tracking: [ticks, heads, H_down, W_down]
    # class_labels: [num_classes]
    image = np.moveaxis(image, 0, -1)  # move C channel to the end
    cmap_viridis = sns.color_palette('viridis', as_cmap=True)
    cmap_spectral = sns.color_palette("Spectral", as_cmap=True)
    figscale = 2
    route_len = predictions.shape[0]

    mpl.rcParams['font.family'] = ['sans-serif']
    mpl.rcParams['font.sans-serif'] = ['IPAMincho', 'DejaVu Sans']

    with tqdm(total=predictions.shape[-1]+1, initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:

        pbar_inner.set_description('Iterating through to build frames')

        frames = []
        route_steps = {}
        route_colours = []

        n_steps = predictions.shape[-1]
        n_heads = attention_tracking.shape[1]
        step_linspace = np.linspace(0, 1, n_steps)
        for stepi in np.arange(0, n_steps, 1):
            pbar_inner.set_description('Making frames for gif')

            attention_now = attention_tracking[stepi] # Make it smooth for pretty

            certainties_now = certainties[1, :stepi+1]
            attention_interp = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), image.shape[:2], mode='bilinear')[0]
            attention_interp = (attention_interp.flatten(1) - attention_interp.flatten(1).min(-1, keepdim=True)[0])/(attention_interp.flatten(1).max(-1, keepdim=True)[0] - attention_interp.flatten(1).min(-1, keepdim=True)[0])
            attention_interp = attention_interp.reshape(n_heads, image.shape[0], image.shape[1])

            colour = list(cmap_spectral(step_linspace[stepi]))
            route_colours.append(colour)
            for headi in range(min(8, n_heads)):
                com_attn = np.copy(attention_interp[headi])
                com_attn[com_attn < np.percentile(com_attn, 97)] = 0.0
                if headi not in route_steps:
                    A = attention_interp[headi].detach().cpu().numpy()
                    centres, areas = find_island_centers(A, threshold=0.7)
                    route_steps[headi] = [centres[np.argmax(areas)]]
                else:
                    A = attention_interp[headi].detach().cpu().numpy()
                    centres, areas = find_island_centers(A, threshold=0.7)
                    route_steps[headi] = route_steps[headi] + [centres[np.argmax(areas)]]

            mosaic = [['head_0', 'head_0_overlay', 'head_1', 'head_1_overlay', 'head_2', 'head_2_overlay', 'head_3', 'head_3_overlay'],
                      ['head_4', 'head_4_overlay', 'head_5', 'head_5_overlay', 'head_6', 'head_6_overlay', 'head_7', 'head_7_overlay'],
                      ['route_0', 'route_1', 'route_2', 'route_3', 'route_4', 'route_5', 'route_6', 'route_7'],
                      ['route_8', 'route_9', 'route_10', 'route_11', 'certainty', 'certainty', 'certainty', 'certainty'],
                      ]


            img_aspect = image.shape[0]/image.shape[1]
            # print(img_aspect)
            aspect_ratio = (len(mosaic[0])*figscale, len(mosaic)*figscale*img_aspect)
            fig, axes = plt.subplot_mosaic(mosaic, figsize=aspect_ratio)
            for ax in axes.values():
                ax.axis('off')


            axes['certainty'].plot(np.arange(len(certainties_now)), certainties_now, 'k-', linewidth=figscale*1, label='1-(normalised entropy)')
            axes['certainty'].plot(len(certainties_now)-1, certainties_now[-1], 'k.', markersize=figscale*4)
            axes['certainty'].axis('off')
            axes['certainty'].set_ylim([-0.05, 1.05])
            axes['certainty'].set_xlim([0, certainties.shape[-1]+1])

            for route_i in range(min(11, route_len)):
                ax_route = axes[f'route_{route_i}']
                target = targets[route_i]
                ps = torch.softmax(torch.from_numpy(predictions[route_i, :, stepi]), -1)
                k = 10 if len(class_labels) > 10 else len(class_labels)
                topk = torch.topk (ps, k, dim = 0, largest=True).indices.detach().cpu().numpy()
                top_classes = np.array(class_labels)[topk]
                true_class = target
                colours = [('r' if ci != true_class else 'g') for ci in topk]
                bar_heights = ps[topk].detach().cpu().numpy()


                ax_route.bar(np.arange(len(bar_heights))[::-1], bar_heights, color=np.array(colours), alpha=1)
                ax_route.set_ylim([0, 1])


                for i, (name) in enumerate(top_classes):
                    prob = ps[i]
                    is_correct = name==class_labels[true_class]
                    fg_color = 'darkgreen' if is_correct else 'crimson'
                    text_str = f'{name[:40]}'
                    ax_route.text(
                        0.05,
                        #0.95 - i * 0.055,  # Adjust vertical position for each line
                        0.95 - i * 0.1,
                        # Adjust vertical position for each line
                        text_str,
                        transform=ax_route.transAxes,
                        verticalalignment='top',
                        fontsize=14,  # Increased font size
                        color=fg_color,
                        alpha=0.8,
                        path_effects=[
                            patheffects.Stroke(linewidth=3, foreground='aliceblue'),
                            patheffects.Normal()
                        ])


            attention_now = attention_tracking[max(0, stepi-5):stepi+1].mean(0)  # Make it smooth for pretty
            # attention_now = (attention_tracking[:stepi+1, 0] * decay).sum(0)/(decay.sum(0))
            certainties_now = certainties[1, :stepi+1]
            attention_interp = torch.nn.functional.interpolate(torch.from_numpy(attention_now).unsqueeze(0), image.shape[:2], mode='nearest')[0]
            attention_interp = (attention_interp.flatten(1) - attention_interp.flatten(1).min(-1, keepdim=True)[0])/(attention_interp.flatten(1).max(-1, keepdim=True)[0] - attention_interp.flatten(1).min(-1, keepdim=True)[0])
            attention_interp = attention_interp.reshape(n_heads, image.shape[0], image.shape[1])

            for hi in range(min(8, n_heads)):
                ax = axes[f'head_{hi}']
                img_to_plot = cmap_viridis(attention_interp[hi].detach().cpu().numpy())
                if img_to_plot.shape[-1] == 3: # RGB
                    ax.imshow(img_to_plot)
                else:
                    ax.imshow(img_to_plot, cmap='gray', vmin=0, vmax=1)

                ax_overlay = axes[f'head_{hi}_overlay']

                these_route_steps = route_steps[hi]
                y_coords, x_coords = zip(*these_route_steps)
                y_coords = image.shape[-2] - np.array(list(y_coords))-1

                if image.shape[-1] == 3: # RGB
                    ax_overlay.imshow(np.flip(image, axis=0), origin='lower')
                else: # Grayscale
                    ax_overlay.imshow(np.flip(image, axis=0), origin='lower', cmap='gray', vmin=0, vmax=1)


                # ax_overlay.imshow(np.flip(image, axis=0), origin='lower')
                # ax.imshow(np.flip(solution_maze, axis=0), origin='lower')
                arrow_scale = 1.5 if image.shape[0] > 32 else 0.8
                for i in range(len(these_route_steps)-1):
                    dx = x_coords[i+1] - x_coords[i]
                    dy = y_coords[i+1] - y_coords[i]

                    ax_overlay.arrow(x_coords[i], y_coords[i], dx, dy, linewidth=1.6*arrow_scale*1.3, head_width=1.9*arrow_scale*1.3, head_length=1.4*arrow_scale*1.45, fc='white', ec='white', length_includes_head = True, alpha=1)
                    ax_overlay.arrow(x_coords[i], y_coords[i], dx, dy, linewidth=1.6*arrow_scale, head_width=1.9*arrow_scale, head_length=1.4*arrow_scale, fc=route_colours[i], ec=route_colours[i], length_includes_head = True)


                ax_overlay.set_xlim([0,image.shape[1]-1])
                ax_overlay.set_ylim([image.shape[0]-1, 0])
                ax_overlay.axis('off')


            fig.tight_layout(pad=0.1)



            canvas = fig.canvas
            canvas.draw()
            image_numpy = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
            image_numpy = (image_numpy.reshape(*reversed(canvas.get_width_height()), 4)[:,:,:3])
            frames.append(image_numpy)
            plt.close(fig)
            pbar_inner.update(1)
        true_decoding = []
        for i in range(route_len):
          text = class_labels[targets[i]]
          if text == "<EOS>":
            continue
          true_decoding.append(text)
        true_decoding = "_".join(true_decoding)
        pbar_inner.set_description('Saving gif')
        filename = f'prediction_{bi:07d}_{index:02d}_{true_decoding}.gif'
        imageio.mimsave(f'{save_location}/{filename}', frames, fps=15, loop=100)
