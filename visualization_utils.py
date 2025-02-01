import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from skimage import filters
from IPython.display import display

def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

def show_image_and_heatmap(heatmap: torch.Tensor, image: Image.Image, relevnace_res: int = 256, interpolation: str = 'bilinear',):
    image = image.resize((relevnace_res, relevnace_res))
    image = np.array(image)
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = heatmap.reshape(1, 1, heatmap.shape[-1], heatmap.shape[-1])
    heatmap = torch.nn.functional.interpolate(heatmap, size=relevnace_res, mode=interpolation)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap.reshape(relevnace_res, relevnace_res).cpu()

    vis = show_cam_on_image(image, heatmap)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    vis = vis.astype(np.uint8)
    vis = Image.fromarray(vis).resize((relevnace_res, relevnace_res))

    return vis

def show_only_heatmap(heatmap: torch.Tensor, relevnace_res: int = 256, interpolation: str = 'bilinear', gassussian_kernel_size: int = 3):
    # Apply gaussian blur to heatmap
    # heatmap = gaussian_blur(heatmap, kernel_size=gassussian_kernel_size)

    heatmap = heatmap.reshape(1, 1, heatmap.shape[-1], heatmap.shape[-1])
    heatmap = torch.nn.functional.interpolate(heatmap, size=relevnace_res, mode=interpolation)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap.reshape(relevnace_res, relevnace_res).cpu()

    vis = heatmap
    vis = np.uint8(255 * vis)

    # Show in black and white
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_GRAY2BGR)

    vis = Image.fromarray(vis).resize((relevnace_res, relevnace_res))

    return vis

import matplotlib.pyplot as plt

def visualize_tokens_attentions(attention, tokens, image, prompt, heatmap_interpolation="nearest", show_on_image=True):
    """
    Visualize token attention maps and optionally save the visualization to a file.

    Args:
        attention (torch.Tensor): Attention tensor of shape (num_tokens, width, height).
        tokens (list): List of token strings.
        image (PIL.Image or ndarray): Original image.
        prompt (str): prompt for the image.
        heatmap_interpolation (str): Interpolation method for heatmaps.
        show_on_image (bool): Whether to overlay heatmaps on the original image.
    """
    token_vis = []
    for j, token in enumerate(tokens):
        if j >= attention.shape[0]:
            break

        if show_on_image:
            vis = show_image_and_heatmap(attention[j], image, relevnace_res=512, interpolation=heatmap_interpolation)
        else:
            vis = show_only_heatmap(attention[j], relevnace_res=512, interpolation=heatmap_interpolation)

        token_vis.append((token, vis))

    # Display the token and the attention map in a grid, with K tokens per row
    K = 4
    n_rows = (len(token_vis) + K - 1) // K  # Ceiling division
    fig, axs = plt.subplots(n_rows, K, figsize=(K * 5, n_rows * 5))

    for i, (token, vis) in enumerate(token_vis):
        row, col = divmod(i, K)
        if n_rows > 1:
            ax = axs[row, col]
        elif K > 1:
            ax = axs[col]
        else:
            ax = axs

        ax.imshow(vis)
        ax.set_title(token)
        ax.axis("off")

    # Hide unused subplots
    for j in range(i + 1, n_rows * K):
        row, col = divmod(j, K)
        if n_rows > 1:
            axs[row, col].axis('off')
        elif K > 1:
            axs[col].axis('off')

    plt.tight_layout()
    output_path = prompt + "_heatmap.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    # plt.show()





    # We want to return the figure so that we can save it to a file
    # return fig

def show_images(images, titles=None, size=1024, max_row_length=5, figsize=None, col_height=10):
    if isinstance(images, Image.Image):
        images = [images]

    if len(images) == 1:
        img = images[0]
        img = img.resize((size, size))
        plt.imshow(img)
        plt.axis('off')

        if titles is not None:
            plt.title(titles[0])
            
        plt.show()
    else:
        images = [img.resize((size, size)) for img in images]

        # Check if the number of titles matches the number of images
        if titles is not None:
            assert len(images) == len(titles), "Number of titles should match the number of images"

        n_images = len(images)
        n_cols = min(n_images, max_row_length)
        n_rows = (n_images + n_cols - 1) // n_cols  # Calculate the number of rows needed

        if figsize is None:
            figsize=(n_cols * col_height, n_rows * col_height)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        # Display images in the subplots
        for i, img in enumerate(images):
            axs[i].imshow(img)
            if titles is not None:
                axs[i].set_title(titles[i])
            axs[i].axis("off")

        # Turn off any unused subplots
        for ax in axs[len(images):]:
            ax.axis("off")

    # plt.show()

