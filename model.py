import torch
import torch.nn as nn

class VAE(nn.Module):

    def __init__(self, config, device):
        super().__init__()

        act_map = {"leaky": nn.LeakyReLU(),
                   "tanh": nn.Tanh(),
                   "none": nn.Identity()}

        # Verify config
        assert config['enc_conv_act'] in act_map, f"[INFO] {config['enc_conv_act']} cannot be found in activation map. Please check."
        assert config['enc_fc_mean_act'] in act_map, f"[INFO] {config['enc_fc_mean_actt']} cannot be found in activation map. Please check."
        assert config['enc_fc_log_var_act'] in act_map, f"[INFO] {config['enc_fc_log_var_act']} cannot be found in activation map. Please check."
        assert config['dec_fc_act'] in act_map, f"[INFO] {config['dec_fc_act']} cannot be found in activation map. Please check."
        assert config['dec_tconv_act'] in act_map, f"[INFO] {config['dec_tconv_act']} cannot be found in activation map. Please check."

        assert config['enc_fc_layers'][-1] == config['dec_fc_layers'][0] == config['latent_dim'], "[INFO] Please check the latent dimensionality."

        # Encoder
        if config['conditional']:
            config['enc_conv_channels'][0] += config['num_classes']

        self.enc_conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = config['enc_conv_channels'][i], out_channels = config['enc_conv_channels'][i+1],
                          kernel_size = config['enc_conv_kernel_size'][i], stride = config['enc_conv_stride'][i]),
                nn.BatchNorm2d(num_features = config['enc_conv_channels'][i+1]),
                act_map[config['enc_conv_act']]
                ) for i in range(len(config['enc_conv_kernel_size']))])

        self.enc_mean_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features = config['enc_fc_layers'][i], out_features = config['enc_fc_layers'][i+1]),
                act_map[config['enc_fc_mean_act']]
                ) for i in range(len(config['enc_fc_layers']) - 1)])

        self.enc_log_var_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features = config['enc_fc_layers'][i], out_features = config['enc_fc_layers'][i+1]),
                act_map[config['enc_fc_log_var_act']]
                ) for i in range(len(config['enc_fc_layers']) - 1)])


        # Decoder
        if config['conditional']:
            config['dec_fc_layers'][0] += config['num_classes']

        self.dec_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features = config['dec_fc_layers'][i], out_features = config['dec_fc_layers'][i+1]),
                act_map[config['dec_fc_act']]
                ) for i in range(len(config['dec_fc_layers']) - 1)])

        self.dec_tconv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels = config['dec_tconv_channels'][i], out_channels = config['dec_tconv_channels'][i+1],
                                   kernel_size = config['dec_tconv_kernel_size'][i], stride = config['dec_tconv_stride'][i]),
                nn.BatchNorm2d(num_features = config['dec_tconv_channels'][i+1]),
                act_map[config['dec_tconv_act']]
                ) for i in range(len(config['dec_tconv_kernel_size']))])

        # Config
        self.config = config
        self.device = device

    def reparmeterize(self, mean, log_var):
        std = torch.exp(log_var) ** 0.5
        z = torch.randn_like(std)
        return mean + std*z

    def generate(self, z, label=None):
        """
            if this is called without using forward, make sure:
                z: torch tensor on same device as model
                label:
                    if conditional: torch.tensor on the same device as model
                    if non-conditional: set to None
        """

        out = z

        if self.config['conditional']:

            assert label is not None, f"[INFO] In conditional VAE, label cannot be {label}."

            label_vector = torch.zeros((z.shape[0], self.config['num_classes']))
            batch_idx = torch.arange(0, z.shape[0])
            label_idx = label[torch.arange(0, z.shape[0])]
            label_vector[batch_idx, label_idx] = 1
            out = torch.cat([out, label_vector.to(self.device)], dim=1)

            torch.cuda.synchronize() ## Synchronize check point 4


        for layer in self.dec_fc:
            out = layer(out)

        hw = torch.tensor (out.shape[1] / self.config['dec_tconv_channels'][0])
        spatial = int(torch.sqrt(hw).item())
        out = out.reshape((z.shape[0], self.config['dec_tconv_channels'][0], spatial, spatial))

        for block in self.dec_tconv_blocks:
            out = block(out)

        return out

    def forward(self, x, label=None):
        """
            x: torch tensor on same device as model
            label:
                if conditional: torch.tensor on the same device as model
                if non-conditional: set to None
        """

        out = x

        # Add conditional information
        if self.config['conditional']:

            assert label is not None, f"[INFO] For conditional VAE, label input cannot be set as {label}."

            label_ch_maps = torch.zeros((x.shape[0], self.config['num_classes'], 28, 28))
            batch_idx = torch.arange(0, x.shape[0])
            label_idx = label[torch.arange(0, x.shape[0])]
            label_ch_maps[batch_idx, label_idx, :, :] = 1
            out = torch.cat([out, label_ch_maps.to(self.device)], dim=1)

            torch.cuda.synchronize() ## Synchronize check point 3


        # Go through encoding conv block
        for block in self.enc_conv_blocks:
            out = block(out)

        # Flatten
        out = out.reshape((x.shape[0], -1))

        # Go through mean fc layers
        out_mean = out
        for layer in self.enc_mean_fc:
            out_mean = layer(out_mean)

        # Go through log var fc layers
        out_log_var = out
        for layer in self.enc_log_var_fc:
            out_log_var = layer(out_log_var)

        # Reparameterization
        z = self.reparmeterize(mean = out_mean, log_var = out_log_var)


        # Go through dec fc layers and transpose conv blocks
        if self.config['conditional']:
            assert label is not None, f"[INFO] For conditional VAE, label input cannot be set as {label}."

        final_out = self.generate(z = z,
                                  label = label)

        return final_out, out_mean, out_log_var

