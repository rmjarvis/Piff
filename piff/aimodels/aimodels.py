class ResidualBlockConv(nn.Module):
    """
    Residual Block with Convolutional layers.
    Maintains the spatial dimensions (padding=1 for kernel_size=3).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class SpatialSoftmax(nn.Module):
    """
    Applies Softmax over the spatial dimensions of the input.
    Assumes input is (Batch, Channels, Height, Width).
    Flattens spatial dims, applies softmax, and reshapes back.
    """
    def forward(self, x):
        b, c, h, w = x.shape
        # Flatten all dimensions except batch
        # Logic matches user's original which softmaxed over 'input_dim' (all pixels)
        # Note: If channels > 1, this softmaxes over all channels*pixels. 
        # Since output is 1 channel, it is just pixels.
        x = x.view(b, -1)
        x = nn.functional.softmax(x, dim=1)
        x = x.view(b, c, h, w)
        return x

class Conv2dAutoEncoder(nn.Module):
    def __init__(self, grid_size: int = 25, latent_dim: int = 3, hidden_channels: int = 32):
        super().__init__()
        
        self.grid_size = grid_size
        
        # --- Encoder ---
        # 1. Conv -> Hidden (25x25)
        # 2. Downsample -> Hidden*2 (12x12)
        # 3. Downsample -> Hidden*4 (6x6)
        # 4. Flatten -> Linear -> Latent
        
        # Flatten size calculation:
        # 6x6 spatial * (hidden_channels*4) channels
        flatten_dim = (hidden_channels * 4) * 6 * 6
        
        self.encoder = nn.Sequential(
            # Input: (B, 1, 25, 25)
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels),
            
            # Downsample 1: 25x25 -> 12x12
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels*2),
            
            # Downsample 2: 12x12 -> 6x6
            nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels*4),
            
            # Flatten and Linear
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dim)
        )

        # --- Decoder ---
        # 1. Linear -> Flattened size
        # 2. Unflatten -> (Hidden*4, 6, 6)
        # 3. Upsample -> Hidden*2 (12x12)
        # 4. Upsample -> Hidden (25x25)
        # 5. Output Conv -> 1 Channel
        # 6. Spatial Softmax
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flatten_dim),
            nn.Unflatten(1, (hidden_channels*4, 6, 6)),
            
            # Upsample 1: 6x6 -> 12x12
            nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels*2),
            
            # Upsample 2: 12x12 -> 25x25
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            ResidualBlockConv(hidden_channels),
            
            # Final reconstruction
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            SpatialSoftmax()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)