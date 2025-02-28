import sys
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
    GradCAMElementWise,
    KPCA_CAM,
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)

sys.path.append("../src/")
from helpers.data_utils import UnlearningDataLoader
UDL = UnlearningDataLoader(
    dataset='tiny-imagenet',
    batch_size=1,
    image_size=64,
    seed=3407,
    is_vit=False,
    is_class_unlearning=True,
    class_to_forget="pizza",
)
dl, _ = UDL.load_data()
num_classes = len(UDL.classes)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument(
        "--image-path", type=str, default="./pizza_image.jpeg", help="Input image path"
    )
    parser.add_argument(
        "--aug-smooth",
        action="store_true",
        help="Apply test time augmentation to smooth the CAM",
    )
    parser.add_argument(
        "--eigen-smooth",
        action="store_true",
        help="Reduce noise by taking the first principle component"
        "of cam_weights*activations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=[
            "gradcam",
            "fem",
            "hirescam",
            "gradcam++",
            "scorecam",
            "xgradcam",
            "ablationcam",
            "eigencam",
            "eigengradcam",
            "layercam",
            "fullgrad",
            "gradcamelementwise",
            "kpcacam",
            "shapleycam",
        ],
        help="CAM method",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Output directory to save the images",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./resnet18.pth",
        help="Path to the model to use",
    )
    args = parser.parse_args()

    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print("Using CPU for computation")

    return args


if __name__ == "__main__":
    """python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        "kpcacam": KPCA_CAM,
    }

    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore

    

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    img_size = 128 
    print(args.image_path)
    rgb_img = cv2.imread(args.image_path, 1)
    rgb_img = cv2.resize(rgb_img, (img_size, img_size))
    rgb_image = rgb_img[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=(122.4786, 114.2755, 101.3963), std=(70.4924, 68.5679, 71.8127)
    ).to(args.device)


    for model_path in os.listdir("gradcams/models"):
        model = torch.load(os.path.join("gradcams/models", model_path)).to(torch.device(args.device)).eval()        
        # target_layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        target_layers = [model.layer4[-1]]

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [ClassifierOutputTarget(16)]
        # print(targets)
        # targets = [ClassifierOutputReST(281)]
        targets = [ClassifierOutputTarget(16)]

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model, target_layers=target_layers) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 1
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth,
            )

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
        gb = gb_model(input_tensor, target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        os.makedirs(args.output_dir, exist_ok=True)

        cam_output_path = os.path.join(args.output_dir, f"{model_path}_{args.method}_cam.jpg")
        # gb_output_path = os.path.join(args.output_dir, f"{args.method}_gb.jpg")
        # cam_gb_output_path = os.path.join(args.output_dir, f"{args.method}_cam_gb.jpg")

        cv2.imwrite(cam_output_path, cam_image)
        # cv2.imwrite(gb_output_path, gb)
        # cv2.imwrite(cam_gb_output_path, cam_gb)

        count = 0
        for x, y in dl["forget"]:
            # if count == 486:
            if count==72:
                x = x.to(args.device)
                y = y.to(args.device)
                output = model(x)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                print(f"model_path {model_path}")
                print(f"Probability of class {y.item()}: {probabilities[0, y].item()}")
                predicted_class = torch.argmax(probabilities, dim=1)
                print(f"Predicted class index: {predicted_class.item()}")
                count += 1
            else:
                count += 1
