import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pix2pix_turbo import Pix2Pix_Turbo  # Assuming Pix2Pix_Turbo is the model class
import torchvision.transforms.functional as F

def process_image_stack(image_stack, model, output_dir, img_name, prompt):
    """
    Process an image stack using the provided model.

    Parameters:
    - image_stack (torch.Tensor): The image stack tensor.
    - model (Pix2Pix_Turbo): The model to use for processing.
    - output_dir (str): Directory to save the output image.
    - img_name (str): The name of the input image file.
    - prompt (str): The prompt to use for processing.

    Returns:
    None
    """
    with torch.no_grad():
        # Generate output image using the model with the given prompt
        output_image = model(image_stack.cuda(), prompt=prompt)
        # Convert output tensor to PIL image
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

        # Save the output image using the original image name
        output_pil.save(os.path.join(output_dir, img_name))

def get_image_names(input_dir, txt_file=None):
    """
    Get the list of image names from the directory or a txt file.

    Parameters:
    - input_dir (str): The root folder containing the dataset.
    - txt_file (str or None): The path to the txt file containing image names, if provided.

    Returns:
    A list of image names.
    """
    if txt_file:
        # Read image names from the txt file
        with open(txt_file, 'r') as f:
            img_names = [line.strip() for line in f if line.strip().endswith('.jpg')]
    else:
        # List image names from the directory
        exposure_folders = ['0.2', '0.5', '0.8']
        img_names = [img for img in os.listdir(os.path.join(input_dir, exposure_folders[0])) if img.endswith('.jpg')]
    return img_names

def get_image_stacks_and_process(input_dir, model, output_dir, prompt, txt_file=None, max_images=None):
    """
    Get image stacks from the input directory and process them.

    Parameters:
    - input_dir (str): The root folder containing the dataset with subfolders for different exposures.
    - model (Pix2Pix_Turbo): The model to use for processing.
    - output_dir (str): Directory to save the output images.
    - prompt (str): The prompt to use for processing.
    - txt_file (str or None): Path to a txt file containing image names, if provided.
    - max_images (int or None): Maximum number of images to process, if provided.

    Returns:
    None
    """
    exposure_folders = ['0.2', '0.5', '0.8']

    # Ensure that the transform is consistent with the image preparation in the main code
    T = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize images to a common size
    ])

    # Get list of images either from the txt file or from the directory
    img_names = get_image_names(input_dir, txt_file)

    # Limit the number of images to process if max_images is set
    if max_images is not None:
        img_names = img_names[:max_images]

    for img_name in tqdm(img_names, desc="Processing image stacks"):
        img_stack = []
        for exposure in exposure_folders:
            img_path = os.path.join(input_dir, exposure, img_name)
            img = Image.open(img_path).convert("RGB")
            img_t = T(img)
            img_t = F.to_tensor(img_t)
            img_stack.append(img_t)
        img_stack = torch.stack(img_stack, dim=0).unsqueeze(0)  # Add batch dimension [1, 3, 3, 512, 512]

        # Process the image stack using the original image name
        process_image_stack(img_stack, model, output_dir, img_name, prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the input directory')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--prompt', type=str, default='exposure control', help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--txt', type=str, default=None, help='path to the txt file containing image names (optional)')
    parser.add_argument('--max_images', type=int, default=None, help='maximum number of images to process (optional)')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()

    # Load and process image stacks from the input directory (or from the txt file if provided)
    get_image_stacks_and_process(args.input_dir, model, args.output_dir, args.prompt, args.txt, args.max_images)