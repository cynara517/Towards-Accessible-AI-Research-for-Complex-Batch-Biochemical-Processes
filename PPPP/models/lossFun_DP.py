import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesGANLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, recon_weight=1.0, adv_weight=1.0, cosine_weight=0.1, use_focal=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.recon_weight = recon_weight
        self.adv_weight = adv_weight
        self.cosine_weight = cosine_weight
        self.use_focal = use_focal

    def forward(self, real_data, disc_real_output, disc_fake_output=None, mode=None, fake_data=None, compute_adv=True):
        if mode == 'generator':
            if fake_data is None:
                raise ValueError("fake_data must be provided in generator mode")
            recon_loss = F.mse_loss(fake_data, real_data)
            if compute_adv:
                bce_loss = F.binary_cross_entropy(disc_fake_output, torch.ones_like(disc_fake_output))
                if self.use_focal:
                    pt = torch.exp(-bce_loss)
                    adv_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                else:
                    adv_loss = bce_loss
            else:
                adv_loss = 0.0
            cosine_loss = 1 - F.cosine_similarity(
                fake_data.flatten(start_dim=1),
                real_data.flatten(start_dim=1)
            ).mean()
            total_loss = (self.recon_weight * recon_loss +
                          self.adv_weight * adv_loss +
                          self.cosine_weight * cosine_loss)
            return total_loss
        elif mode == 'discriminator':
            real_loss = F.binary_cross_entropy(disc_real_output, torch.ones_like(disc_real_output))
            fake_loss = F.binary_cross_entropy(disc_fake_output, torch.zeros_like(disc_fake_output))
            return (real_loss + fake_loss) / 2
