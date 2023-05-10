#!/usr/bin/env python
import argparse

import torch
from torchvision import transforms

from datasets import Fitzpatrick17kDataset, ISICDataset
from models import Generator
from utils import kid_ebpe

class Int8GeneratorWrapper(torch.nn.Module):
    """Wrapper for a generator that operates on images in the range [-1,1].
    This wrapper rescales in the inputs and outputs such that they are within
    the range (0,255). The latent vector is returned.

    original generator:
      input range: -1 to 1
      output range: -1 to 1
    wrapped generator:
      input range: 0 to 255
      output range: 0 to 255"""
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, y):
        x = x*(1/122.5)-1
        img, latent = self.generator(x, y)
        out = torch.clamp((img+1)*122.5, min=0, max=255).to(torch.int32)
        return out

class DetectorWrapper(torch.nn.Module):
    """Wrap the detector so that it doesn't break with PyTorch 1.9."""
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, x):
        old = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
        features = self.detector(x, return_features=True)
        torch._C._jit_override_can_fuse_on_gpu(old)
        return features

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--dataset", type=str, choices=["f17k", "isic"])
    parser.add_argument("--detector_path", type=str, 
            default="/projects/leelab3/derm/inception_pretrained/inception-2015-12-05.pt")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    detector_path = args.detector_path
    if args.dataset.lower() == 'f17k':
        dataset_class = Fitzpatrick17kDataset
    elif args.dataset.lower() == 'isic':
        dataset_class = ISICDataset
    else:
        raise ValueError("Invalid dataset {:s}".format(args.dataset))
    im_size = args.image_size

    # Other defaults
    device = 'cuda'
    batch_size = 4

    # Load generator model
    generator = Generator(im_size=im_size)
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    generator = Int8GeneratorWrapper(generator)

    # Load the detector model (pretrained Inception)
    detector = torch.jit.load(detector_path)
    detector = DetectorWrapper(detector)

    # Convert from (0,1) range to (0, 255) range
    normalize = transforms.Normalize(mean=0,
                                     std=1/255)
    transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            normalize])
    dataset = dataset_class(transform=transform)
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=True, 
            num_workers=4)

    print(kid_ebpe(dataloader, generator, detector, max_images=4000))

if __name__ == "__main__":
    main()
