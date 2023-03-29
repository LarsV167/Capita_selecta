import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn import UpsamplingNearest2d
from torch.nn.utils import spectral_norm
from torch.distributions import Normal
import torch
from pathlib import Path
from torch import nn
import pdb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#import u_net
import utils
import glob
from torchvision.io import read_image
import os
from torchvision.models import vgg16
import torchvision.transforms as T
import torch
import os
import vae_spade
from torch.utils.data import DataLoader
import numpy as np
#import GAN_Lars
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np


path_data=r"C:\Users\20191679\Documents\Master\CS_image_analysis\TrainingData"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device) #Device used: using GPU will significantly speed up the training.


random.seed(42)
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

#Save results to tensorboard
TENSORBOARD_LOGDIR = "VAE_runs"



# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 1 #I think we wanted to generate images on whole dataset so this should be 0?
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 25
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
Z_DIM = 256


# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )



patients = [
    path
    for path in glob.glob(path_data+r"\p*[0-9]")
]

random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}


dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)


train_features, train_labels = next(iter(dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")



img = train_features[0].squeeze()
label = train_labels[0].squeeze()
plt.imshow(img, cmap="gray")
plt.imshow(label,cmap='gray',alpha=0.3)
plt.show()
print(f"Label: {label}")
print(label.size(dim=1))


# create new segmentations with background values
label_bg=np.zeros(np.shape(label))
for i in range(64):
    for j in range(64):
        if label[i,j] == 0:
            label_bg[i,j] = img[i,j]
            
        else:
            label_bg[i,j] = torch.max(img[:,:])


plt.imshow(label_bg[:,:], cmap='gray')
plt.show()

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)


val_features, val_labels = next(iter(valid_dataloader))


# initialise model, optimiser
vae_model = vae_spade.VAE()
optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)
# add a learning rate scheduler based on the lr_lambda function
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



 # training loop
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    for x_real, _ in tqdm(dataloader, position=0):
        # needed to zero gradients in each iteration
        optimizer.zero_grad()
        x_recon, mu, logvar = vae_model(x_real, _)  # forward pass
        loss = vae_spade.vae_loss(x_real, x_recon, mu, logvar)
        current_train_loss += loss.item()
        loss.backward()  # backpropagate loss
        optimizer.step()  # update weights

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)

    scheduler.step()  # step the learning step scheduler

    # save examples of real/fake images
    if (epoch + 1) % 1 == 0:
        
        vae_model.eval()
        img_grid = make_grid(
            torch.cat((x_recon[:5], x_real[:5])), nrow=5, padding=12, pad_value=-1
        )
        writer.add_image(
           "Real/fake_recon",
            np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5,
            epoch + 1,
        )

        noise = torch.randn(32, Z_DIM)
        image_samples = vae_model.generator(noise, _)
        img_grid = make_grid(
            torch.cat((image_samples[:5], image_samples[5:])),
            nrow=5,
            padding=12,
            pad_value=-1,
        )
        writer.add_image(
            "Samples",
            np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5,
            epoch + 1,
        )

    vae_model.train()
    

torch.save(vae_model.state_dict(), CHECKPOINTS_DIR / "vae_model.pth")


for (num,_) in enumerate(img_grid):
    print(num)
    img = img_grid[num].squeeze()
    #label = train_labels[num].squeeze()
    plt.imshow(img, cmap="gray")
    #plt.imshow(label,cmap='gray',alpha=0.3)
    plt.show()























