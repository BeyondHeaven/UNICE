import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from thop import profile

def adjust_exposure(image_tensor, exposure):
    mean_img = image_tensor.mean().item()
    factor = exposure / mean_img
    adjusted_image = image_tensor * factor
    return adjusted_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the input directory')
    parser.add_argument('--filelist', type=str, help='path to a text file containing a list of image filenames')
    parser.add_argument('--prompt', type=str, default='exposure control', help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--exposure', type=float, required=True, help='desired exposure level for the images')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()

    # process a single image
    filename = os.listdir(args.input_dir)[0]
    input_image = Image.open(os.path.join(args.input_dir, filename)).convert('RGB')
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    # translate the image
    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
        c_t = adjust_exposure(c_t, args.exposure)
        output_image = model(c_t, args.prompt)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # save the output image
    output_pil.save(os.path.join(args.output_dir, filename))

    # calculate FLOPS
    flops, params = profile(model, inputs=(c_t, args.prompt))
    print(f"FLOPS: {flops}, Parameters: {params}")