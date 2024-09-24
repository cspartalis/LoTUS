import os
import os
from tqdm import tqdm
from datasets import load_dataset


def download_imagenet(download_dir):
    # Ensure the directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Download the training set
    print("Downloading training set...")
    train_dataset = load_dataset("imagenet-1k", split="train", cache_dir=download_dir)
    train_dataset.save_to_disk(os.path.join(download_dir, "imagenet1k_train"))

    # Download the validation set
    print("Downloading validation set...")
    val_dataset = load_dataset(
        "imagenet-1k", split="validation", cache_dir=download_dir
    )
    val_dataset.save_to_disk(os.path.join(download_dir, "imagenet1k_val"))

    print(f"Download complete. Dataset saved in {download_dir}")


def save_imagenet_splits(dataset_dict, base_output_dir="~/data/ImageNet1k"):
    base_output_dir = os.path.expanduser(base_output_dir)

    def save_images(dataset, split_name):
        split_dir = os.path.join(base_output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for idx, example in enumerate(tqdm(dataset, desc=f"Saving {split_name} images")):
            image = example['image']
            label = example['label']

            # Create class directory
            class_dir = os.path.join(split_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)

            # Convert image to RGB if it's in RGBA mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            # Save the image
            image_path = os.path.join(class_dir, f"{idx}.jpg")
            try:
                image.save(image_path, "JPEG")
            except Exception as e:
                print(f"Error saving image {idx} in {split_name} split: {str(e)}")
                continue

    for split_name, dataset in dataset_dict.items():
        print(f"Processing {split_name} split...")
        save_images(dataset, split_name)

    print(f"All splits have been saved to {base_output_dir}")


if __name__ == "__main__":
    # download_dir = ("~/data/imagenet-1k")
    # download_imagenet(download_dir)
    dataset = load_dataset("imagenet-1k", cache_dir="~/data/imagenet-1k/")
    save_imagenet_splits(dataset)
