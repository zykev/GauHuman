import torch
import torch.nn as nn
from utils.sh_utils import eval_sh_bases,augm_rots


class MLP_decoder(nn.Module):
    def __init__(self, feature_out_dim):
        super().__init__()
        self.output_dim = feature_out_dim
        #self.fc1 = nn.Linear(16, 128).cuda()
        #self.fc1 = nn.Linear(128, 128).cuda()
        #self.fc2 = nn.Linear(128, self.output_dim).cuda()

        #self.fc0 = nn.Linear(16, 16).cuda()
        #self.fc1 = nn.Linear(16, 32).cuda()
        #self.fc2 = nn.Linear(32, 64).cuda()
        #self.fc3 = nn.Linear(128, 128).cuda()
        self.fc4 = nn.Linear(128, 256).cuda()

    def forward(self, x):
        input_dim, h, w = x.shape
        x = x.permute(1,2,0).contiguous().view(-1, input_dim) #(16,48,64)->(48,64,16)->(48*64,16)
        #x = torch.relu(self.fc0(x))
        #x = torch.relu(self.fc1(x))
        #x = self.fc2(x)
        #x = torch.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(h, w, self.output_dim).permute(2, 0, 1).contiguous()
        return x


class CNN_decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.input_dim = input_dim
        #self.output_dim = output_dim

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()


    def forward(self, x):
        
        x = self.conv(x)

        return x
    

class OpacityRefiner(nn.Module):
    def __init__(self, semantic_feature_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(semantic_feature_dim, output_dim)

    def forward(self, opacity, semantic_feature):
        x = nn.functional.sigmoid(self.fc(semantic_feature))
        x = x + opacity
        x = nn.functional.sigmoid(x) 

        return x 


class ColorMLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ColorMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out
        self.n_neurons, self.n_hidden_layers = 256, 4

        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            setattr(self, "linear" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, x):

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "linear" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x
    

class ColorRefiner(nn.Module):
    def __init__(self, sh_feature_dim, semantic_feature_dim):
        super().__init__()
        d_in = sh_feature_dim * 3

        # self.use_normal = False
        self.sh_degree = 3
        self.cano_view_dir = True

        # if self.use_normal:
        #     d_in += 3 # quasi-normal by smallest eigenvector...
        # if self.sh_degree > 0:
        d_in += (self.sh_degree + 1) ** 2 - 1 # sh_feature_dim - 1
            # self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        d_in += semantic_feature_dim

        d_out = 3
        self.mlp = ColorMLP(d_in, d_out)
        self.color_activation = nn.Sigmoid()

    def compose_input(self, gaussians, camera, transforms):
        # features = gaussians.get_features.squeeze(-1) # sh coefficients
        features = gaussians.get_features.view(-1, (self.sh_degree + 1) ** 2 * 3)
        n_points = features.shape[0]
        # if self.use_normal:
        #     scale = gaussians._scaling
        #     rot = build_rotation(gaussians._rotation)
        #     normal = torch.gather(rot, dim=2, index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        #     features = torch.cat([features, normal], dim=1)
        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            if self.cano_view_dir:
                # T_fwd = gaussians.fwd_transform
                # R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                R_bwd = transforms.transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = 45
                if self.training and view_noise_scale > 0.:
                    view_noise = torch.tensor(augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                                              dtype=torch.float32,
                                              device=dir_pp.device).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, view_noise)
            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            # dir_embed = self.sh_embed(dir_pp_normalized)
            dir_embed = eval_sh_bases(self.sh_degree, dir_pp_normalized)[..., 1:]
            features = torch.cat([features, dir_embed], dim=1) # base sh functions values

        features = torch.cat([features, gaussians.get_semantic_feature.squeeze(1)], dim=1)

        return features


    def forward(self, gaussians, camera, transforms):
        inp = self.compose_input(gaussians, camera, transforms)
        output = self.mlp(inp)
        color = self.color_activation(output)
        return color
