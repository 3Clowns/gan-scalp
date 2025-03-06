import torch
import torch.nn as nn
import time
import numpy as np
import tqdm
import wandb
from rnngan.utils.constants import LATENT_DIM, DEVICE
def train_gan(generator, discriminator, dataloader, epochs, lr=0.0002):
    """
    Trains a Generative Adversarial Network (GAN).

    Args:
        generator (nn.Module): The generator model that creates fake data.
        discriminator (nn.Module): The discriminator model that distinguishes real data from fake data.
        dataloader (DataLoader): DataLoader providing real data for training.
        epochs (int): Number of training epochs.
        lr (float, optional): Learning rate for optimizers. Defaults to 0.0002.
    """
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    d_loss_history = []
    g_loss_history = []
    criterion = nn.BCELoss()
    st = 0
    fn = 0
    import tqdm  # Ensure tqdm is imported if not already

    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        st = time.time()
        epoch_d_loss = []
        epoch_g_loss = []
        for real_data in tqdm.tqdm(dataloader, leave=False, desc="Batches"):
            start = time.time()
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
            real_data = real_data.unsqueeze(-1).to(DEVICE)

            # Train Discriminator
            opt_d.zero_grad()
            real_outputs = discriminator(real_data)
            d_real_loss = criterion(real_outputs, real_labels)

            z = torch.randn(batch_size, real_data.size(1), generator.input_dim).to(DEVICE)

            fake_data = generator(z)
            #print(fake_data.shape)
            fake_outputs = discriminator(fake_data.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)
            print(d_fake_loss)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            fake_outputs = discriminator(fake_data)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            opt_g.step()
            wandb.log({
                "d_loss_batch": d_loss.item(),
                "g_loss_batch": g_loss.item()
            })
            d_loss_history.append(d_loss)
            g_loss_history.append(g_loss)
            epoch_d_loss.append(d_loss)
            epoch_g_loss.append(g_loss)
        fn = time.time() - st
        print(f'Epoch {epoch} done in {fn} seconds. D_loss {d_loss_history[-1]:.4f}. G_loss {g_loss_history[-1]:.4f}')
        wandb.log({
            "d_loss_epoch": sum(epoch_d_loss)/len(epoch_d_loss),
            "g_loss_epoch": sum(epoch_g_loss)/len(epoch_g_loss),
            "epoch_time": time.time() - st
        })
    checkpoint = {
        "epoch": epochs,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "d_loss_history": d_loss_history,
        "g_loss_history": g_loss_history
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epochs}.pt")
    print(f"Saved checkpoint for  {epochs} epochs")