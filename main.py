import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loss.VGGLoss import VGGLoss
from model import Generator, Discriminator
from dataset.DIV2K import DIV2K

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    DATA_DIR = '/Users/muhammadarham/Drive/MLProjects/SRGAN/data/DIV2K/DIV2K_train_HR/DIV2K_train_HR'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IN_CHANNELS = 3
    NUM_CHANNELS = 64
    NUM_BLOCKS = 16
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NUM_EPOCHS = 5

    dataset = DIV2K(img_dir=DATA_DIR)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    discriminator = Discriminator(in_channels=IN_CHANNELS).to(DEVICE)
    generator = Generator(
        in_channels=IN_CHANNELS,
        num_channels=NUM_CHANNELS,
        num_blocks=NUM_BLOCKS

    ).to(DEVICE)

    vggLoss = VGGLoss(device=DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()

    gen_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter("./runs/losses")

    step = 0
    for epoch in range(NUM_EPOCHS):

        torch.save(generator.state_dict(), 'runs/models/gen_latest.pth')
        torch.save(discriminator.state_dict(), 'runs/models/disc_latest.pth')

        for idx, batch in enumerate(tqdm(train_dataloader)):
            low_res, high_res = batch
            low_res = low_res.to(DEVICE)
            high_res = high_res.to(DEVICE)

            # -- Train Discriminator -- #
            fake = generator(low_res)
            disc_real = discriminator(high_res)
            disc_fake = discriminator(fake.detach())

            disc_loss_real = bce_loss(
                disc_real, torch.ones_like(disc_real)
            )

            disc_loss_fake = bce_loss(
                disc_fake, torch.zeros_like(disc_fake)
            )

            disc_loss = disc_loss_fake + disc_loss_real

            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()

            # -- Train Generator -- #
            disc_fake = discriminator(fake)
            gan_loss = bce_loss(disc_fake, torch.ones_like(disc_fake))
            content_loss = vggLoss(fake, high_res)
            gen_loss = 0.006 * content_loss + 1e-3 * gan_loss

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            writer.add_scalar('Generator Loss / Iteration',
                              gen_loss.item(), step)
            writer.add_scalar('Discriminator Loss / Iteration',
                              disc_loss.item(), step)

            step += 1

        writer.add_scalar(
            'Generator Loss / Epoch', gen_loss.item(), epoch
        )
        writer.add_scalar(
            'Discriminator Loss / Epoch', disc_loss.item(), epoch
        )

        print(f'EPOCH {epoch + 1} / {NUM_EPOCHS}: Generator Loss: {gen_loss.item():.4f} -- Discriminator Loss: {disc_loss.item():.4f}')
