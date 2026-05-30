import torchvision.transforms as T
from config import CFG

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

    
def build_cpu_training_transform(image_size: int) -> T.Compose:
    return T.Compose([
    T.RandomCrop(image_size, pad_if_needed=True),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([
        T.ColorJitter(0.2, 0.2, 0.2, 0.05)
    ], p=0.8),
    T.RandomApply([
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0)),
    ], p=0.2),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_cpu_testing_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
