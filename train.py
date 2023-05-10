#!/usr/bin/env python
'''
Train a generative model to create counterfactuals, using the methodology of 
``Explanation by Progressive Exaggeration'' [1]. Note that the model 
architecture has been modified from [1], and that the loss terms match those 
described in [1], which differs slightly from the original Tensorflow 
implementation of Explanation by Progressive Exaggeration.

[1] Singla, S.; Pollack, B.; Chen, J.; & Batmanghelich, K. Explanation by 
  Progressive Exaggeration. ICLR 2020



This code is a PyTorch re-implementation (with modifications) of the original 
TensorFlow version of Explanation of Progressive Exaggeration, and is provided 
under the following license:

MIT License

Copyright (c) 2019 Sumedha Singla and Kayhan Batmanghelich
Copyright (c) 2023 Alex DeGrave (modifications)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from loss import *
from models import Generator
from models import Discriminator 
from models import DeepDermClassifier 
from models import ModelDermClassifier 
from models import ScanomaClassifier 
from models import SSCDClassifier 
from models import SIIMISICClassifier
from datasets import ISICDataset, Fitzpatrick17kDataset

def main():
    lambda_cycle = 3
    lambda_gan = 1
    lambda_cls = 1
    n_epochs = 500
    n_classes = 10
    min_pred = 0
    max_pred = 1
    training_ratio = 5
    device = 'cuda'
    #device = 'cpu'
    batch_size = 4
    accumulate_steps = 8 
    save_path = 'checkpoint.pth'
    load_checkpoint = False
    save_interval = 100
    datasetclass = ISICDataset
    classifier = DeepDermClassifier()
    #classifier = ModelDermClassifier()
    #classifier = ScanomaClassifier()
    #classifier = SSCDClassifier()
    #classifier = SIIMISICClassifier()

    writer = SummaryWriter(comment='deepderm;isic;lambda_cycle=3')

    im_size = classifier.image_size
    positive_index = classifier.positive_index
    # Images must be in the range (-1, 1)
    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    transform = transforms.Compose([
            transforms.Resize(int(im_size*1.2)),
            transforms.RandomCrop(im_size),
            transforms.ColorJitter(0.2,0,0,0),
            transforms.ToTensor(),
            normalize])
    dataset = datasetclass(transform=transform)
    # mini-batches will only be of size "batch_size", but we load multiple
    # mini-batches on each call to dataloader.__next__ for assistance with
    # accumulating gradients
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size*accumulate_steps, 
            shuffle=True, 
            drop_last=True, 
            persistent_workers=True, 
            num_workers=2)

    bins = np.linspace(min_pred, max_pred, n_classes+1)
    bin_centers = np.array([bins[i:i+2].mean() for i in range(n_classes)])
    bins = torch.Tensor(bins)
    bin_centers = torch.Tensor(bin_centers)

    # initialize models
    generator = Generator(im_size=im_size)
    discriminator = Discriminator()
    classifier.eval()

    # send to device (cuda)
    generator.to(device)
    discriminator.to(device)
    classifier.to(device)
    bins = bins.to(device)
    bin_centers = bin_centers.to(device)

    # set up losses
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    discriminator_loss = DiscriminatorLoss()
    generator_loss = GeneratorLoss()
    klloss = KLLoss()
    discriminator_loss.to(device)
    generator_loss.to(device)
    l1loss.to(device)
    l2loss.to(device)
    klloss.to(device)

    # optimizers
    g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.9))

    # load if necessary
    start_epoch = 0
    if load_checkpoint:
        checkpoint = torch.load(save_path)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_opt.load_state_dict(checkpoint['g_opt'])
        d_opt.load_state_dict(checkpoint['d_opt'])
        start_epoch = checkpoint['epoch'] + 1

    # main training loop
    offset = torch.ones(1, dtype=torch.long).to(device)
    for i_epoch in range(start_epoch, n_epochs):
        pbar = tqdm(dataloader)
        nbatches = len(pbar)
        for i_batch, batch in enumerate(pbar):
            im_batch, label_batch = batch
            im_batch = im_batch.to(device) # im_batch is batch_size*accumulate_steps in size

            #### discriminator training ####
            d_opt.zero_grad()
            d_loss_accum = 0
            for i_step in range(accumulate_steps):
                # Here we use "y" to refer to the CLASSIFIER's labels. This
                # technique doesn't use the original labels at all.
                im = im_batch[batch_size*i_step:batch_size*(1+i_step)]
                with torch.no_grad():
                    y_orig = classifier(im)[:,positive_index]
                    y_orig_binned = torch.clamp(torch.bucketize(y_orig, bins)-offset, min=0, max=n_classes-1)

                    # generate images in random target bins 
                    y_target = torch.randint(low=0, high=n_classes, size=(im.shape[0],)).to(device)
                    im_target, _ = generator(im, y_target)

                # discriminator losses
                real_logits = discriminator(im, y_orig_binned)
                discriminator.turn_off_sn()
                fake_logits = discriminator(im_target, y_target)
                discriminator.turn_on_sn()
                d_loss = discriminator_loss(real_logits, fake_logits)*lambda_gan
                d_loss /= accumulate_steps
                d_loss.backward()
                d_loss_accum += d_loss.detach()
            # update parameters
            d_opt.step()
            writer.add_scalar('loss/d', d_loss_accum, i_batch+i_epoch*nbatches)

            #### generator training ####
            if (i_batch+1) % training_ratio == 0:
                g_opt.zero_grad()
                g_loss_accum = 0
                g_loss_gan_accum = 0
                recons_classifier_loss_accum = 0
                altered_classifier_loss_accum = 0
                recons_loss_accum = 0
                cycle_loss_accum = 0

                for i_step in range(accumulate_steps):
                    im = im_batch[batch_size*i_step:batch_size*(1+i_step)]
                    y_orig = classifier(im)[:,positive_index]
                    y_orig_binned = torch.clamp(torch.bucketize(y_orig, bins)-offset, 
                                                min=0, max=n_classes-1)

                    # reconstruct original image
                    im_recons, im_recons_emb = generator(im, y_orig_binned)

                    # generate altered version of images
                    y_target = torch.randint(low=0, high=n_classes, 
                                             size=(im.shape[0],)).to(device)
                    im_target, _ = generator(im, y_target)

                    # cycle the images: im --> im_target --> im_cycle
                    im_cycle, im_cycle_emb = generator(im_target, y_orig_binned)

                    # reconstruction losses
                    recons_loss = l1loss(im_recons, im)*lambda_cycle
                    cycle_loss = l1loss(im_cycle, im)*lambda_cycle

                    # classifier-consistency losses
                    y_recons = classifier(im_recons)[:,positive_index]
                    recons_classifier_loss = klloss(y_orig, y_recons)*lambda_cls

                    desired_pred = bin_centers[y_target]
                    y_altered = classifier(im_target)[:,positive_index]
                    altered_classifier_loss = klloss(desired_pred, y_altered)*lambda_cls

                    # GAN loss
                    fake_logits = discriminator(im_target, y_target)
                    g_loss_gan = generator_loss(fake_logits)*lambda_gan

                    # combine loss; lambda applied above
                    g_loss = g_loss_gan + recons_loss + cycle_loss + altered_classifier_loss + recons_classifier_loss
                    g_loss /= accumulate_steps
                    g_loss.backward()

                    # for logging only
                    g_loss_accum += g_loss.cpu().detach()
                    g_loss_gan_accum += g_loss_gan.cpu().detach()/accumulate_steps
                    recons_loss_accum += recons_loss.cpu().detach()/accumulate_steps
                    cycle_loss_accum += cycle_loss.cpu().detach()/accumulate_steps
                    altered_classifier_loss_accum += altered_classifier_loss.cpu().detach()/accumulate_steps
                    recons_classifier_loss_accum += recons_classifier_loss.cpu().detach()/accumulate_steps

                g_opt.step()
                pbar.set_description(
                      "{:05d}-{:05d} |".format(i_epoch, i_batch) +\
                      "d: {:06.3f} |".format(d_loss_accum.detach().cpu().numpy()) +\
                      "g: {:06.2f} |".format(g_loss_accum.detach().cpu().numpy()) +\
                      "g_gan: {:06.2f} |".format(g_loss_gan_accum.detach().cpu().numpy()) +\
                      "recons: {:05.3f} |".format(recons_loss_accum.detach().cpu().numpy()) +\
                      "cycle: {:05.3f} |".format(cycle_loss_accum.detach().cpu().numpy()) +\
                      "target_classification: {:05.3f} |".format(altered_classifier_loss_accum.detach().cpu().numpy()) +\
                      "recons_classification: {:05.3f} |".format(recons_classifier_loss_accum.detach().cpu().numpy())
                      )
                writer.add_scalar('loss/g', g_loss_accum, i_batch+i_epoch*nbatches)
                writer.add_scalar('loss/g_gan', g_loss_gan_accum, i_batch+i_epoch*nbatches)
                writer.add_scalar('loss/recons', recons_loss_accum, i_batch+i_epoch*nbatches)
                writer.add_scalar('loss/cycle', cycle_loss_accum, i_batch+i_epoch*nbatches)
                writer.add_scalar('loss/target_cls', altered_classifier_loss_accum, i_batch+i_epoch*nbatches)
                writer.add_scalar('loss/recons_cls', recons_classifier_loss_accum, i_batch+i_epoch*nbatches)
        # save every epoch
        torch.save({'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'g_opt': g_opt.state_dict(),
                    'epoch': i_epoch},
                    save_path)
        # Save backups every `save_interval` epochs
        if i_epoch != 0 and i_epoch % save_interval == 0:
            torch.save({'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'd_opt': d_opt.state_dict(),
                        'g_opt': g_opt.state_dict(),
                        'epoch': i_epoch},
                        save_path+".{:d}".format(i_epoch))

if __name__ == "__main__":
    main()
