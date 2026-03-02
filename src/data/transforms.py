"""Image transforms for DINO training."""

from torchvision import transforms


def get_default_transforms(
    image_size: int = 224,
    is_train: bool = True
) -> transforms.Compose:
    """
    Get default image transforms for DINO.
    
    Args:
        image_size: Target image size
        is_train: Whether to use training transforms (with augmentation)
        
    Returns:
        Composed transform
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return transform
