from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision.transforms as transforms
import cv2
import glob
import os
import numpy as np
import sys
from PIL import Image
from ..model.model import MidasNet_small
from ..utils.customization_utils import replace_dw_layers, replace_relu6_with_hardtanh
import matplotlib.pyplot as plt

def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)

def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """

    sample = image.to(device).unsqueeze(0)
    model = model.to(device)

    if optimize and device == torch.device("cuda"):

        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    else:
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def run(input_path, output_path, model, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        side (bool): RGB and depth side by side in output images?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
        grayscale (bool): use a grayscale colormap?
    """
    print("Initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    """
    model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                            non_negative=True, blocks={'expand': True})"""
    net_w, net_h = 256, 256
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to w=256, h=256
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    if optimize and (device == torch.device("cuda")):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    # get input
    if input_path is not None:
        image_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(image_names)
    else:
        print("No input path specified. Grabbing images from camera.")

    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    print("Start processing")

    if input_path is not None:
        if output_path is None:
            print("Warning: No output path specified. Images will be processed but not shown or stored anywhere.")
        for index, image_name in enumerate(image_names):

            print("  Processing {} ({}/{})".format(image_name, index + 1, num_images))

            im = Image.open(image_name)
            imshape = im.size + (3,)
            image = transform(im)

            # compute
            with torch.no_grad():
                prediction = process(device, model, model_type, image, (net_w, net_h), imshape[1::-1],
                                     optimize, False)

            # output
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
                )
                if not side:
                    write_depth(filename, prediction, grayscale, bits=2)
                else:
                    raise Exception
                write_pfm(filename + ".pfm", prediction.astype(np.float32))


def get_device(conf):
    os.environ['CUDA_HOME'] = conf.CUDA_HOME

    if torch.cuda.is_available() :
        device = torch.device('cuda')
        os.environ['CUDA_HOME'] = conf.CUDA_HOME
    else:
        device = torch.device('cpu')
    print(f"Testing on {device} device.")
    return device


def quantize(target, quant_mode, quant_dir, model, img, device, deploy_check):
    quantizer = torch_quantizer(quant_mode, model, (img), output_dir = quant_dir, device=device, 
                                target=target)
    qmodel = quantizer.quant_model
    _ = qmodel(img)
    quantizer.export_quant_config()
    quantizer.export_xmodel(quant_dir, deploy_check=deploy_check)
    quantizer.export_torch_script(output_dir = quant_dir)


def get_model(conf):

    device = get_device(conf)

    model = MidasNet_small(align_corners=False)
    parameters = torch.load(conf.MODEL_WEIGHTS, 
                            map_location=torch.device('cpu'))

    if "optimizer" in parameters:
        parameters = parameters["model"]

    model.load_state_dict(parameters)
    _ = model(torch.randn((1,3,conf.IMG_H,conf.IMG_W)))
    replace_relu6_with_hardtanh(model)
    model = model.eval()
    model.to(device)
    model = replace_dw_layers(model)
    return model,device

def get_midas_transform():
    transform = transforms.Compose([
    transforms.Resize((192, 256)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    return transform


def infer_sample(model, transform, image_path='/workspace/demodel/dog.jpg', depth_map_path = '/workspace/demodel/dog_inferred_depth.jpg'):
    img = transform(Image.open(image_path)).unsqueeze(0).to(model.device)
    o = model(img).squeeze().detach().cpu().numpy()
    min_val = np.min(o)
    max_val = np.max(o)
    normalized_image = ((o - min_val) / (max_val - min_val)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    plt.imsave(depth_map_path, normalized_image, cmap='gray')
