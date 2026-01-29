import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), # normalize the data -> how many standard deviations
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), # apply another conv (while keeping channel dim) to enrich the output.
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # convolve 256 -> 256 (no pooling)
        x1 = self.enc1(x)
        # down 256 -> 128
        x2 = self.enc2(self.pool(x1))
        # down 128 -> 64
        x3 = self.enc3(self.pool(x2))
        # down 64 -> 32
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4

class TransformerBottleneck(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=4, spatial_res=32):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, spatial_res**2, dim)) # learnable pos embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # init with normal dist

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True, dropout=0.1, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # 'num_layers' size transformer

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # squashes the input into a vector to feed the transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, C, H, W) # return to image form
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # up 32 -> 64
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = ConvBlock(512, 256)
        # up 64 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = ConvBlock(256, 128)
        # up 128 -> 256
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = ConvBlock(128, 64)
        # the predicted map 256 -> 256
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, bottleneck_out, x3, x2, x1):
        x = self.up1(bottleneck_out)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return self.out(x)

class TumorDetector(nn.Module):
    def __init__(self, img_size=256, u_depth=3):
        super().__init__()
        self.encoder = Encoder()
        self.transformer = TransformerBottleneck(dim=512, spatial_res=img_size // 2**u_depth)
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        bottleneck_out = self.transformer(x4)
        logits = self.decoder(bottleneck_out, x3, x2, x1)
        return logits

    def predict(self, x, decision_thresh=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            prediction = (probs > decision_thresh).float()
        return prediction, logits

class TumorLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_score(self, batch_logits, batch_gt_masks, eps=1e-6):
        probs = torch.sigmoid(batch_logits)
        prediction = (probs > 0.5).float()
        intersection = (prediction * batch_gt_masks).sum() # |A cross B|
        union = prediction.sum() + batch_gt_masks.sum() # |A u B|
        return (2 * intersection + eps) / (union + eps)

    def dice_loss(self, batch_logits, batch_gt_masks, eps=1e-6):
        probs = torch.sigmoid(batch_logits)
        intersection = (probs * batch_gt_masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + batch_gt_masks.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (union + eps)
        return 1.0 - dice.mean() # the closer the mean dice is to 1.0 the better

    def forward(self, batch_logits, batch_gt_masks):
        return self.dice_weight * self.dice_loss(batch_logits, batch_gt_masks) + self.bce_weight * self.bce(batch_logits, batch_gt_masks)