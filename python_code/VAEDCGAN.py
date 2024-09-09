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
from matplotlib import animation
from IPython.display import HTML
from torchvision.utils import save_image
import torchvision

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
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
        #output["combined_map"] = self.apply_transform(nmap)
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

class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None):
        super(VAEEncoder, self).__init__()

        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # Create the encoder using the modules list
        self.encoder = nn.Sequential(*modules)

        # Calculate the flattened size for the linear layers
        self.flatten_size = hidden_dims[-1] * 4 * 4  # Adjust this based on your conv layer output

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]


class VAEDecoder(nn.Module):
    def __init__(self, feature_dim, out_channels):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(feature_dim, 256 * 4 * 4)

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape to a 4D tensor

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # Use sigmoid for the last layer to output values between 0 and 1
        return x
    
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super(DCGANGenerator, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4) 
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Changed in_channels and out_channels
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Tanh for normalizing the output to [-1, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to (batch_size, channels, height, width)
        return self.model(x)

class DCGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(DCGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # This layer should output a 1x1 spatial dimension
            nn.Flatten(),  # Flatten the 1x1 output
            nn.Sigmoid()  # Output the probability
        )

    def forward(self, x):
        return self.model(x)
    
class VAE_GAN(nn.Module):
    def __init__(self, vae_encoder, vae_decoder, dcgan_generator, dcgan_discriminator):
        super(VAE_GAN, self).__init__()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.dcgan_generator = dcgan_generator
        self.dcgan_discriminator = dcgan_discriminator
        
        # Initialize lists to track losses
        self.gen_losses = []
        self.dis_losses = []
        self.vae_losses = []
        self.dgz_values = []
        self.gen_grid_imgs = []

    def reparameterize(self, mu, logvar):
        """Apply the reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # VAE Encoding: Encode the input to latent space representation
        mu, logvar = self.vae_encoder(x)
        z = self.reparameterize(mu, logvar)

        # DCGAN Generation: Generate images from the latent space representation
        generated_images = self.dcgan_generator(z)

        # DCGAN Discrimination: Discriminate the generated images
        real_or_fake = self.dcgan_discriminator(generated_images)

        return generated_images, real_or_fake, mu, logvar
    
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
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=2)
        return train_dataloader, val_dataloader

def validate_vae(dataloader, model, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    criterion_recon = nn.MSELoss()  # Reconstruction loss
    criterion_kl = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    with torch.no_grad():
        for data in dataloader:
            real_images = data["dmap"].to(device)
            mu, logvar = model.vae_encoder(real_images)
            z = model.reparameterize(mu, logvar)
            recon_images = model.vae_decoder(z)

            recon_loss = criterion_recon(recon_images, real_images)
            kl_loss = criterion_kl(mu, logvar)
            total_loss += recon_loss.item() + kl_loss.item()

    return total_loss / len(dataloader)

def validate_dcgan(dataloader, model, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    criterion_adv = nn.BCELoss()

    with torch.no_grad():
        for data in dataloader:
            real_images = data["delta_vmap"].to(device)
            # Generate fake images
            z = torch.randn(real_images.size(0), 64, device=device)
            fake_images = model.dcgan_generator(z)

            # Compute loss for real and fake images
            real_or_fake = model.dcgan_discriminator(fake_images).view(-1, 1)
            real_labels = torch.ones(real_images.size(0), 1, device=device)
            fake_labels = torch.zeros(real_images.size(0), 1, device=device)
            real_loss = criterion_adv(real_or_fake, real_labels)
            fake_loss = criterion_adv(real_or_fake, fake_labels)
            total_loss += real_loss.item() + fake_loss.item()

    return total_loss / len(dataloader)

def train_vae(dataloader, vae_gan, epochs, device):
    vae_optimizer = optim.Adam(list(vae_gan.vae_encoder.parameters()) + list(vae_gan.vae_decoder.parameters()), lr=0.0002)

    criterion_recon = nn.MSELoss()  # Reconstruction loss
    criterion_kl = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence

    vae_gan.train()
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            real_images = data["dmap"].to(device)  # Adjust according to your data structure
            vae_optimizer.zero_grad()

            # VAE Forward pass
            mu, logvar = vae_gan.vae_encoder(real_images)
            z = vae_gan.reparameterize(mu, logvar)
            recon_images = vae_gan.vae_decoder(z)

            # Compute loss
            recon_loss = criterion_recon(recon_images, real_images)
            kl_loss = criterion_kl(mu, logvar)
            vae_loss = recon_loss + kl_loss

            # Backward pass and optimization
            vae_loss.backward()
            vae_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], "
                      f"VAE Loss: {vae_loss.item()}")
                # print("VAE input data shape:", real_images.shape)
                # print("VAE output shape:", mu.shape)
    
    torch.save(vae_gan.vae_encoder.state_dict(), 'vae_encoder.pth')
    torch.save(vae_gan.vae_decoder.state_dict(), 'vae_decoder.pth')

def train_dcgan(vae_data_loader, dcgan_data_loader, vae_gan, epochs, device):
    vae_gan.vae_encoder.load_state_dict(torch.load('vae_encoder.pth'))
    vae_gan.vae_encoder.eval()  # Set to evaluation mode
    dcgan_optimizer = optim.Adam(list(vae_gan.dcgan_generator.parameters()) + list(vae_gan.dcgan_discriminator.parameters()), lr=0.01)
    criterion_adv = nn.BCELoss()  # Adversarial loss

    vae_gan.train()
    for epoch in range(epochs):
        for i, dcgan_data in enumerate(dcgan_data_loader):
            real_images = dcgan_data["delta_vmap"].to(device)  # Adjust according to your data structure
            batch_size = real_images.size(0)

            dcgan_optimizer.zero_grad()

            # Use trained VAE encoder to generate features
            with torch.no_grad():
                mu, _ = vae_gan.vae_encoder(real_images)
                z = vae_gan.reparameterize(mu, torch.zeros_like(mu))

            # DCGAN Forward pass
            fake_images = vae_gan.dcgan_generator(z)
            real_or_fake = vae_gan.dcgan_discriminator(fake_images).view(-1, 1)

            # Compute loss
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            real_loss = criterion_adv(real_or_fake, real_labels)
            fake_loss = criterion_adv(real_or_fake, fake_labels)
            dcgan_loss = real_loss + fake_loss
            dgz = vae_gan.dcgan_discriminator(fake_images).mean().item()
            vae_gan.dgz_values.append(dgz)
            dgz = real_or_fake.mean().item()

            # Backward pass and optimization
            dcgan_loss.backward()
            dcgan_optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dcgan_data_loader)}], "
                      f"DCGAN Loss: {dcgan_loss.item()}, D(G(z)): {dgz}")
                # print("DCGAN input data shape:", real_images.shape)


                with torch.no_grad():
                    grid = torchvision.utils.make_grid(fake_images.cpu(), normalize=True)
                    vae_gan.gen_grid_imgs.append(grid)
                save_path = f"output_images/epoch_{epoch+1}_step_{i+1}.png"
                save_image(grid, save_path, normalize=True)

def show_reconstructed_images(vae_gan, dataloader, device):
    vae_gan.vae_encoder.load_state_dict(torch.load('vae_encoder.pth'))
    vae_gan.vae_decoder.load_state_dict(torch.load('vae_decoder.pth'))
    vae_gan.vae_encoder.eval()
    vae_gan.vae_decoder.eval()

    # Get a batch of images
    data_batch = next(iter(dataloader))
    images = data_batch["dmap"].to(device)  # Use the correct key for your data

    # Reconstruct images
    with torch.no_grad():
        mu, logvar = vae_gan.vae_encoder(images)
        z = vae_gan.reparameterize(mu, logvar)
        recon_images = vae_gan.vae_decoder(z)

    # Convert images for visualization
    images = images.cpu()
    recon_images = recon_images.cpu()

    # Display original and reconstructed images
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(images[i].permute(1, 2, 0), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon_images[i].permute(1, 2, 0), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original Images')
    axes[1, 0].set_title('Reconstructed Images')
    plt.show()




class VisualizeModel:
    def __init__(self, trainer):
        self.trainer = trainer

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.trainer.gen_losses, label="Generator")
        plt.plot(self.trainer.dis_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def plot_dgz(self):
        plt.figure(figsize=(10, 5))
        plt.title("D(G(z)) During Training")
        plt.plot(self.trainer.dgz_values, label="D(G(z))")
        plt.xlabel("Iterations")
        plt.ylabel("D(G(z)) Value")
        plt.legend()
        plt.show()

    def show_generated_images(self, epoch):
        image_file = f"{epoch}_output.png"
        img = plt.imread(image_file)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.show()
        
    def gen_output(self):
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i.numpy(), (1, 2, 0)), animated=True)] for i in self.trainer.gen_grid_imgs]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        HTML(ani.to_jshtml())
        plt.show()
        ani.save("idc_vae_dcgan_animation.gif", writer="pillow", fps=1)

def main():
    # Initialize model components
    vae_encoder = VAEEncoder(in_channels=1, feature_dim=64).to(device)
    vae_decoder = VAEDecoder(feature_dim=64, out_channels=1).to(device)
    dcgan_generator = DCGANGenerator(latent_dim=64, out_channels=1).to(device)
    dcgan_discriminator = DCGANDiscriminator(in_channels=1).to(device)

    # Initialize combined model
    vae_gan = VAE_GAN(vae_encoder, vae_decoder, dcgan_generator, dcgan_discriminator).to(device)

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for each channel (R, G, B)
    ])

    # Define the paths to your VAE and DCGAN data
    vae_dataroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vae_data")
    dcgan_dataroot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dcgan_data")

    # Prepare data loaders
    vae_train_dataloader, _ = Data(dataroot=vae_dataroot, pattern="d_*.npz", transform=transform).dataset_prep()
    dcgan_train_dataloader, _ = Data(dataroot=dcgan_dataroot, pattern="d_*.npz", transform=transform).dataset_prep()


    # Stage 1: Train VAE
    print("Starting VAE training.")
    train_vae(vae_train_dataloader, vae_gan, epochs=10, device=device)
    show_reconstructed_images(vae_gan, vae_train_dataloader, device)
    
    # # Stage 2: Train DCGAN
    # print("Starting DCGAN training.")
    # train_dcgan(vae_train_dataloader, dcgan_train_dataloader, vae_gan, epochs=2, device=device)
    
    # # Visualization (optional)
    # visualizer = VisualizeModel(vae_gan)
    # visualizer.plot_losses()
    # visualizer.plot_dgz()
    # visualizer.gen_output()

if __name__ == "__main__":
    main()
