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

def normalize_gradcam(heatmap1, heatmap2, heatmap3):
    """
    Normalizes two Grad-CAM heatmaps based on their combined min and max values.
    """
    global_min = np.min([heatmap1.min(), heatmap2.min(), heatmap3.min()])
    global_max = np.max([heatmap1.max(), heatmap2.max(), heatmap3.max()])
    denom = (global_max - global_min)  # adding a small constant to avoid division by zero

    # Normalize the heatmaps
    numer1 = heatmap1 - global_min
    heatmap1_normalized = numer1 / denom

    numer2 = heatmap2 - global_min
    heatmap2_normalized = numer2 / denom

    numer3 = heatmap3 - global_min
    heatmap3_normalized = numer3 / denom

    return heatmap1_normalized, heatmap2_normalized, heatmap3_normalized


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

    img_size = 128 
    print(args.image_path)
    rgb_img = cv2.imread(args.image_path, 1)
    rgb_img = cv2.resize(rgb_img, (img_size, img_size))
    rgb_image = rgb_img[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=(122.4786, 114.2755, 101.3963), std=(70.4924, 68.5679, 71.8127)
    ).to(args.device)

    instance_model = torch.load(os.path.join("gradcams/models", "tiny-imagenet-gold-instance.pth")).to(torch.device(args.device)).eval()
    class_model = torch.load(os.path.join("gradcams/models", "tiny-imagenet-gold-class.pth")).to(torch.device(args.device)).eval()
    original_model = torch.load(os.path.join("gradcams/models", "tiny-imagenet-original.pth")).to(torch.device(args.device)).eval()
    target_layers_instance = [instance_model.layer4]
    target_layers_class = [class_model.layer4]
    target_layers_original = [original_model.layer4]
    targets = [ClassifierOutputTarget(16)]
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=instance_model, target_layers=target_layers_instance) as cam_instance:
        cam_instance.batch_size = 1
        grayscale_cam_instance = cam_instance(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )
        grayscale_cam_instance = grayscale_cam_instance[0, :]

    with cam_algorithm(model=class_model, target_layers=target_layers_class) as cam_class:
        cam_class.batch_size = 1
        grayscale_cam_class = cam_class(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )
        grayscale_cam_class = grayscale_cam_class[0, :]

    with cam_algorithm(model=original_model, target_layers=target_layers_original) as cam_original:
        cam_original.batch_size = 1
        grayscale_cam_original = cam_original(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )
        grayscale_cam_original = grayscale_cam_original[0, :]

    # grayscale_cam_instance, grayscale_cam_class, grayscale_cam_original = normalize_gradcam(
    #     grayscale_cam_instance, grayscale_cam_class, grayscale_cam_original
    # )

    ############################## 

    cam_image_instance = show_cam_on_image(rgb_img, grayscale_cam_instance, use_rgb=True)
    cam_image_instance = cv2.cvtColor(cam_image_instance, cv2.COLOR_RGB2BGR)

    cam_image_class = show_cam_on_image(rgb_img, grayscale_cam_class, use_rgb=True)
    cam_image_class = cv2.cvtColor(cam_image_class, cv2.COLOR_RGB2BGR)

    cam_image_original = show_cam_on_image(rgb_img, grayscale_cam_original, use_rgb=True)
    cam_image_original = cv2.cvtColor(cam_image_original, cv2.COLOR_RGB2BGR)

    ##############################

    gb_model_instance = GuidedBackpropReLUModel(model=instance_model, device=args.device)
    gb_instance = gb_model_instance(input_tensor, target_category=16)

    cam_mask_instance = cv2.merge([grayscale_cam_instance, grayscale_cam_instance, grayscale_cam_instance])
    cam_gb_instance = deprocess_image(cam_mask_instance * gb_instance)
    gb_instance = deprocess_image(gb_instance)

    ##############################

    gb_model_class = GuidedBackpropReLUModel(model=class_model, device=args.device)
    gb_class = gb_model_class(input_tensor, target_category=16)

    cam_mask_class = cv2.merge([grayscale_cam_class, grayscale_cam_class, grayscale_cam_class])
    cam_gb_class = deprocess_image(cam_mask_class * gb_class)
    gb_class = deprocess_image(gb_class)

    ##############################

    gb_model_original = GuidedBackpropReLUModel(model=original_model, device=args.device)
    gb_original = gb_model_original(input_tensor, target_category=16)

    cam_mask_original = cv2.merge([grayscale_cam_original, grayscale_cam_original, grayscale_cam_original])
    cam_gb_original = deprocess_image(cam_mask_original * gb_original)
    gb_original = deprocess_image(gb_original)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path_instance = os.path.join(args.output_dir, "gradcam++_instance_cam.jpg")
    cam_output_path_class = os.path.join(args.output_dir, "gradcam++_class_cam.jpg")
    cam_output_path_original = os.path.join(args.output_dir, "gradcam++_original_cam.jpg")

    cv2.imwrite(cam_output_path_instance, cam_image_instance)
    cv2.imwrite(cam_output_path_class, cam_image_class)
    cv2.imwrite(cam_output_path_original, cam_image_original)

    # count = 0
    # for x, y in dl["forget"]:
    #     if count==72:
    #         x = x.to(args.device)
    #         y = y.to(args.device)
    #         output = model(x)
    #         probabilities = torch.nn.functional.softmax(output, dim=1)
    #         print(f"model_path {model_path}")
    #         print(f"Probability of class {y.item()}: {probabilities[0, y].item()}")
    #         predicted_class = torch.argmax(probabilities, dim=1)
    #         print(f"Predicted class index: {predicted_class.item()}")
    #         count += 1
    #     else:
    #         count += 1
