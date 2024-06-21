from denoising_util import *
import DataMilking
class Ximg_to_Ypdf_Autoencoder(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super(Ximg_to_Ypdf_Autoencoder, self).__init__()
        
        # Create encoder based on the provided layer configuration
        encoder_modules = []
        for layer in encoder_layers:
            encoder_modules.append(layer)
            if isinstance(layer, nn.Conv2d):
                encoder_modules.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Create decoder based on the provided layer configuration
        decoder_modules = []
        for layer in decoder_layers:
            decoder_modules.append(layer)
            if isinstance(layer, nn.ConvTranspose2d):
                decoder_modules.append(nn.ReLU())
        
        # Replace the last ReLU with Sigmoid for the final layer
        if isinstance(decoder_modules[-1], nn.ReLU):
            decoder_modules[-1] = nn.Sigmoid()
        
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def trainer(self, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, checkpoints_enabled = True, num_epochs=10):
        #come back here...
        for epoch in range(num_epochs):
            for data in dataloader:
                img, _ = data
                img = img.to(device)
                # ===================forward=====================
                output = self(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item()}')
        torch.save(self.state_dict(), 'autoencoder.pth')

# Example usage
encoder_layers = [
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
]

decoder_layers = [
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
]

autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)