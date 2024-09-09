import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils
import torch.nn.functional as F
from matplotlib import animation
from IPython.display import HTML

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set the device based on CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class for image data
class CustomImageDataset(Dataset):
    """A custom dataset class for loading image data."""

    def __init__(self, path, pattern, transform=None):
        self.file_list = glob.glob(os.path.join(path, pattern))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        output = self.process_data(data)
        return output

    def process_data(self, data):
        """Process and transform the data into the desired format."""
        output = {}
        output["delta_vmap"] = self.transform_data(data['delta_vmap'], (1, 64, 64))
        output["dI"] = torch.tensor(data["dI"], dtype=torch.float)

        # Process and threshold dmap
        dmap = self.threshold_dmap(data['dmap'], threshold=300)
        output["dmap"] = self.transform_data(dmap, (1, 64, 64))

        nmap = torch.tensor(data['nmap'], dtype=torch.float).permute(2, 0, 1)
        output["combined_map"] = self.apply_transform(nmap)
        return output

    def threshold_dmap(self, dmap, threshold):
        """Apply threshold to dmap."""
        dmap[dmap > threshold] = 0
        return dmap

    def transform_data(self, data, shape):
        """Transform data into a tensor of the specified shape."""
        return torch.tensor(data, dtype=torch.float).reshape(shape)

    def apply_transform(self, data):
        """Apply transformations if any."""
        if self.transform:
            return self.transform(data)
        return data

class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None, **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        # Build Encoder
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,  # Ensure this matches the number of channels in the input images
                    kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
class Data:
    def __init__(self, dataroot, pattern, transform):
        self.dataroot = dataroot
        self.pattern = pattern
        self.transform = transform

    def dataset_prep(self):
        dataset = CustomImageDataset(path=self.dataroot, pattern=self.pattern, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=2)
        return train_dataloader, val_dataloader

def validate_vae(dataloader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            real_images = data["dmap"].to(device)
            recon_images, _, mu, logvar = model(real_images)
            recon_loss = F.mse_loss(recon_images, real_images, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss += recon_loss.item() + kld_loss.item()
    return total_loss / len(dataloader)

def train_vae(dataloader, val_dataloader, model, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, data in enumerate(dataloader):
            real_images = data["dmap"].to(device)
            optimizer.zero_grad()
            recon_images, _, mu, logvar = model(real_images)
            recon_loss = F.mse_loss(recon_images, real_images, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1/100 * kld_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(dataloader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = validate_vae(val_dataloader, model, device)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
    # Save the trained VAE components
    torch.save(model.encoder.state_dict(), 'vae_encoder.pth')
    torch.save(model.decoder.state_dict(), 'vae_decoder.pth')
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def show_reconstructed_images(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        data_batch = next(iter(dataloader))
        images = data_batch["dmap"].to(device)
        recon_images, _, _, _ = model(images)

        images = images.cpu()
        recon_images = recon_images.cpu()

        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            axes[0, i].imshow(images[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon_images[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')

        axes[0, 0].set_title('Original Images')
        axes[1, 0].set_title('Reconstructed Images')
        plt.show()

def hyperparameter_tuning(train_dataloader, val_dataloader, epochs, device):
    learning_rates = np.logspace(-8, -1, num=10)
    batch_sizes = [16, 32, 64]

    best_val_loss = float('inf')
    best_lr = None
    best_batch_size = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with learning rate: {lr} and batch size: {batch_size}")

            # Initialize model and optimizer with current hyperparameters
            model = VanillaVAE(in_channels=1, latent_dim=64).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Adjust data loaders for new batch size
            # Note: You'll need to recreate the dataloaders inside this loop
            # with the current batch size

            # Train and validate the model
            train_vae(train_dataloader, val_dataloader, model, epochs, device)

            # Calculate validation loss
            current_val_loss = validate_vae(val_dataloader, model, device)

            # Update best hyperparameters if current validation loss is lower
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_lr = lr
                best_batch_size = batch_size
                print(f"New best model found: lr={lr}, batch_size={batch_size}, val_loss={current_val_loss}")

    print(f"Best Hyperparameters: lr={best_lr}, batch_size={best_batch_size}")
    return best_lr, best_batch_size
    
def main():
    # Initialize model components
    vae_model = VanillaVAE(in_channels=1, latent_dim=64).to(device)

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Define the paths to your VAE and DCGAN data
    vae_dataroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vae_data")

    # Prepare data loaders
    vae_train_dataloader, vae_val_dataloader = Data(dataroot=vae_dataroot, pattern="d_*.npz", transform=transform).dataset_prep()
    
    #best_lr, best_batch_size = hyperparameter_tuning(vae_train_dataloader, vae_val_dataloader, epochs=5, device=device)

    # Stage 1: Train VAE
    print("Starting VAE training.")
    train_vae(vae_train_dataloader, vae_val_dataloader, vae_model, epochs=10, device=device)
    show_reconstructed_images(vae_model, vae_train_dataloader, device)    

if __name__ == "__main__":
    main()