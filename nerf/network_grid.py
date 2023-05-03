import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp, biased_softplus
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance


def build_triangles(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = (
        faces
        + (torch.arange(bs, dtype=torch.int32).to(device=vertices.device) * nv)[
            :, None, None
        ]
    )
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]


def cal_sdf(verts, faces, points):
    # functions modified from ICON

    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]

    Bsize = points.shape[0]

    triangles = build_triangles(verts, faces)
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)

    # closest_triangles = torch.gather(
    #     triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)
    # ).view(-1, 3, 3)
    residues = residues.to(device=points.device)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1)


class MLP(nn.Module):
    def __init__(
        self, dim_in, dim_out, dim_hidden, num_layers, bias=True, activation=None
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        net = []
        for l in range(num_layers):
            net.append(
                nn.Linear(
                    self.dim_in if l == 0 else self.dim_hidden,
                    self.dim_out if l == num_layers - 1 else self.dim_hidden,
                    bias=bias,
                )
            )

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.activation == "relu":
                    x = F.relu(x, inplace=True)
                else:
                    x = torch.tanh(x)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=32,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep')

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        if self.opt.normal_net:
            self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True, activation="relu")

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else biased_softplus
        self.beta = opt.beta

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    # add a density blob to the scene center
    @torch.no_grad()
    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        # g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))
        g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g
    
    @torch.no_grad()
    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def common_forward(self, x):

        # sigma
        enc = self.encoder(x, bound=self.bound)

        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    def sdf_guide_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        sdf_guide = (
            cal_sdf(self.vt, self.faces.unsqueeze(0), x.unsqueeze(0))
            .squeeze(0)
            .squeeze(-1)
        )

        alpha = 1.0 / self.beta
        variable = -1.0 * torch.abs(sdf_guide) * alpha
        sigma_guide = alpha * torch.sigmoid(variable)
        softplus_guide = torch.log(torch.exp(sigma_guide) - 1)
        density_guide = torch.clamp(softplus_guide, min=0.0)

        enc = self.encoder(x, bound=self.bound)
        h = self.sigma_net(enc)

        sigma = nn.functional.softplus(h[..., 0] + density_guide + self.gaussian(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo, enc
    
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        if self.opt.flame:
            dx_pos, _, _ = self.sdf_guide_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dx_neg, _, _ = self.sdf_guide_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dy_pos, _, _ = self.sdf_guide_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dy_neg, _, _ = self.sdf_guide_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dz_pos, _, _ = self.sdf_guide_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
            dz_neg, _, _ = self.sdf_guide_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        else:
            dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
            dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
            dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def _forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if self.opt.flame:
            sigma, albedo, enc = self.sdf_guide_forward(x)
        else:
            sigma, albedo = self.common_forward(x)

        if shading == 'albedo':
            if self.opt.normal_net and self.opt.lambda_normal_consistency > 0:
                normal = self.normal_net(enc)
            else:
                normal = None
            color = albedo
        
        else: # lambertian shading

            if self.opt.normal_net:
                normal = self.normal_net(enc)
            else:
                normal = self.finite_difference_normal(x)
            normal = safe_normalize(normal)
            normal = torch.nan_to_num(normal)

            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        if self.opt.flame:
            sigma, albedo, _ = self.sdf_guide_forward(x)
        else:
            sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10 * self.opt.lr_x},
            {'params': self.sigma_net.parameters(), 'lr': lr * self.opt.lr_x},
        ]        
        
        if self.opt.normal_net:
            params.append({"params": self.normal_net.parameters(), "lr": lr})

        if self.opt.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        if self.opt.dmtet:
            params.append({'params': self.sdf, 'lr': lr * 10})
            params.append({'params': self.deform, 'lr': lr * 10})

        return params