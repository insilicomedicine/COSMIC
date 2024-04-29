from typing import Union

import torch
from torch import nn

from random import shuffle

from ..model_parts import ConfEncoder, ConfDiscriminator, ConfGenerator, \
    LatentDiscriminator

from torch_geometric.data import Batch, Data

from energy_minimization.utils import add_prop, reconstruction_loss, \
    compute_alligment, get_energy, set_conformer


class COSMIC(nn.Module):
    def __init__(self,
                 # common parameters
                 latent_size: int,
                 node_hidden_size: int,
                 edge_hidden_size: int,
                 num_backbone_layers: int,
                 num_main_layers: int,
                 num_refiner_steps: int,
                 num_warmup_iteration: int,
                 num_gaussians: int,
                 lambda_cosloss: float,
                 lambda_mxloss: float,
                 conditions: str,

                 # WGAN relevant parameters
                 use_wgan_part: bool,
                 wgan_lambda_gp: float,
                 wgan_discr_coeff: float,
                 wgan_energy_loss_coeff: float,
                 wgan_num_discr_inter: int,
                 wgan_num_mols_energy: int,

                 # AAE/VAE relevant parameters
                 ae_part_type: str,
                 ae_num_encoder_layers: int,
                 aae_discr_coeff: float,
                 aae_num_discriminator_layers: int,
                 vae_kl_beta: float,

                 # other
                 **kwargs):
        super(COSMIC, self).__init__()

        self.latent_size = latent_size
        self.wgan_lambda_gp = wgan_lambda_gp
        self.aae_discr_coeff = aae_discr_coeff
        self.wgan_discr_coeff = wgan_discr_coeff
        self.wgan_num_mols_energy = wgan_num_mols_energy
        self.wgan_energy_loss_coeff = wgan_energy_loss_coeff

        self.lambda_cosloss = lambda_cosloss
        self.lambda_mxloss = lambda_mxloss
        
        self.vae_kl_beta = vae_kl_beta

        self.num_warmup_iteration = num_warmup_iteration

        self.use_wgan_part = use_wgan_part
        self.ae_part_type = ae_part_type

        self.generator = ConfGenerator(
            latent_size=latent_size,
            num_refiner_steps=num_refiner_steps,
            conditions=conditions,
            node_hidden_size=node_hidden_size,
            edge_hidden_size=edge_hidden_size,
            num_main_layers=num_main_layers,
            num_backbone_layers=num_backbone_layers)

        if self.ae_part_type != 'none':
            self.encoder = ConfEncoder(
                latent_size=(2 if ae_part_type == 'vae' else 1) * latent_size,
                num_backbone_layers=num_backbone_layers,
                num_encoder_layers=ae_num_encoder_layers,
                node_hidden_size=node_hidden_size,
                edge_hidden_size=edge_hidden_size,
                conditions=conditions,
                num_gaussians=num_gaussians,
                use_instance_norm=True
            )
            if self.ae_part_type == 'aae':
                self.aae_discriminator = LatentDiscriminator(
                    latent_size=latent_size,
                    num_backbone_layers=num_backbone_layers,
                    num_discriminator_layers=aae_num_discriminator_layers,
                    node_hidden_size=node_hidden_size,
                    edge_hidden_size=edge_hidden_size,
                    conditions=conditions)

        if self.use_wgan_part:
            self.gan_discriminator = ConfDiscriminator(
                node_size=node_hidden_size,
                edge_size=edge_hidden_size,
                n_interactions=wgan_num_discr_inter,
                num_gaussians=num_gaussians,
                conditions=conditions)

    def sample(self, batch: Union[Data, Batch]):
        add_prop(batch, 'latents',
                 torch.randn(batch.x.shape[0], self.latent_size).to(
                     batch.x.device))

        return self.generator.forward(batch)[0]

    def compute_gradient_penalty(self,
                                 batch,
                                 real_cartesians,
                                 fake_cartesians):
        """Calculates the gradient penalty loss between alligned positions for WGAN GP"""
        graph_nodes_num = [data.cartesian_y.shape[0] for data in
                           batch.to_data_list()]
        splited_real_pos = torch.split(real_cartesians, graph_nodes_num)
        splited_fake_pos = torch.split(fake_cartesians, graph_nodes_num)

        interpolates = []

        for real_pos, fake_pos in zip(splited_real_pos, splited_fake_pos):
            real_pos, fake_pos_aligned = compute_alligment(real_pos, fake_pos)
            alpha = torch.rand((1, 1), device=batch.x.device)

            interpolates.append(
                alpha * real_pos + (1 - alpha) * fake_pos_aligned)

        # Get random interpolation between real and fake samples
        interpolates = torch.cat(interpolates, dim=0).requires_grad_(True)
        d_interpolates = self.gan_discriminator(batch,
                                                cartesian_coords=interpolates)[
            0]

        fake = torch.ones_like(d_interpolates, dtype=torch.float32,
                               device=batch.x.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradients_splited = torch.split(gradients, graph_nodes_num)

        gradient_penalties = [((gradient.norm(2) - 1) ** 2)
                              for gradient in gradients_splited]

        gradient_penalty = torch.stack(gradient_penalties).mean()

        return gradient_penalty

    def compute_loss(self, batch: Union[Data, Batch], batch_idx, current_epoch,
                     optimizer_idx: int = 0):
        if current_epoch > 0:
            sh_lr = 1.0
        else:
            sh_lr = min(1.0, batch_idx / self.num_warmup_iteration)

        loss = 0
        log_dict = dict()

        if optimizer_idx == 0:
            if self.ae_part_type != 'none':
                if self.ae_part_type == 'vae':
                    mu, logvar = torch.split(self.encoder(batch),
                                             self.latent_size, -1)

                    add_prop(batch, 'mu', mu)
                    add_prop(batch, 'logvar', logvar)

                    latents = torch.randn_like(logvar) * torch.exp(
                        0.5 * logvar) + mu
                elif self.ae_part_type == 'aae':
                    latents = self.encoder(batch)

                add_prop(batch, 'latents', latents)

                log_dict['stats_latents_mean_norm'] = torch.norm(
                    latents.mean(dim=0))
                log_dict['stats_latents_std'] = torch.std(latents, dim=0).mean()

                cartesian_pred, torsion_pred = self.generator.forward(batch)

                loss_dist_mx, loss_cos = reconstruction_loss(batch,
                                                             torsion_pred,
                                                             cartesian_pred)

                log_dict['rec_dist_mx'] = loss_dist_mx
                log_dict['rec_cos'] = loss_cos
                
                loss += self.lambda_mxloss * loss_dist_mx + self.lambda_cosloss * loss_cos

                if self.ae_part_type == 'vae':
                    losses_kl = []
                    for data in batch.to_data_list():
                        kl = -0.5 * torch.sum(1 +
                                              data.logvar -
                                              data.mu ** 2 -
                                              data.logvar.exp(), dim=-1).mean()
                    losses_kl.append(kl)
                    loss_kl = torch.stack(losses_kl).mean()

                    log_dict['latent_loss_vae'] = loss_kl
                    loss += self.vae_kl_beta * sh_lr * loss_kl
                elif self.ae_part_type == 'aae':
                    aae_discriminator_out = self.aae_discriminator.forward(
                        batch)
                    aae_discriminator_loss = nn.BCEWithLogitsLoss()(
                        aae_discriminator_out,
                        torch.ones_like(aae_discriminator_out))

                    log_dict['latent_loss_aae'] = aae_discriminator_loss
                    loss += self.aae_discr_coeff * sh_lr * aae_discriminator_loss

            if self.use_wgan_part:
                add_prop(batch, 'latents', torch.randn(batch.x.shape[0],
                                                       self.latent_size).to(
                    batch.x.device))
                cartesian_pred, _ = self.generator(batch)

                wgan_loss = -torch.mean(
                    self.gan_discriminator(
                        batch,
                        cartesian_coords=cartesian_pred
                    )[0]
                )
                log_dict['wgan_G_loss'] = wgan_loss

                loss += self.wgan_discr_coeff * wgan_loss

            log_dict['loss_G'] = loss
        elif optimizer_idx == 1:
            if self.ae_part_type == 'aae':
                with torch.no_grad():
                    latents = self.encoder(batch)

                add_prop(batch, 'latents', latents)
                real_discriminator_out = self.aae_discriminator.forward(batch)
                real_loss = nn.BCEWithLogitsLoss()(real_discriminator_out,
                                                   torch.zeros_like(
                                                       real_discriminator_out))

                add_prop(batch, 'latents', torch.randn_like(latents))
                fake_discriminator_out = self.aae_discriminator.forward(batch)
                fake_loss = nn.BCEWithLogitsLoss()(fake_discriminator_out,
                                                   torch.ones_like(
                                                       fake_discriminator_out))

                aae_D_loss = 0.5 * (real_loss + fake_loss)

                log_dict['aae_D_loss'] = aae_D_loss
                loss += aae_D_loss
            if self.use_wgan_part:
                cartesian_true = batch.cartesian_y

                latents = torch.randn(batch.x.shape[0],
                                      self.latent_size).to(batch.x.device)
                add_prop(batch, 'latents', latents)
                with torch.no_grad():
                    cartesian_pred, torsion_preds = self.generator(batch)

                add_prop(batch, 'cartesian_pred', cartesian_pred)

                fake_validity, fake_pred_energies = self.gan_discriminator(
                    batch,
                    cartesian_coords=cartesian_pred.detach()
                )
                fake_mean = torch.mean(fake_validity)

                real_validity, real_pred_energies = self.gan_discriminator(
                    batch,
                    cartesian_coords=cartesian_true.detach()
                )
                real_mean = torch.mean(real_validity)

                loss_D = -real_mean + fake_mean

                log_dict['wgan_D_loss'] = loss_D
                loss += loss_D

                loss_gp = self.compute_gradient_penalty(
                    batch,
                    cartesian_true.detach(),
                    cartesian_pred.detach())

                log_dict['wgan_D_gp_loss'] = loss_gp
                loss += self.wgan_lambda_gp * loss_gp

                real_true_energies = []
                fake_true_energies = []

                idxs_to_compute_energies = list(range(len(batch.mol)))
                shuffle(idxs_to_compute_energies)
                idxs_to_compute_energies = idxs_to_compute_energies[
                                           :self.wgan_num_mols_energy]
                data_list = batch.to_data_list()
                for i in idxs_to_compute_energies:
                    d = data_list[i]
                    real_true_energies.append(
                        get_energy(d.mol, normalize=True, addHs=True))
                    fake_true_energies.append(
                        get_energy(set_conformer(d.mol, d.cartesian_pred),
                                   normalize=True, addHs=True))

                fake_pred_energies = fake_pred_energies / batch.num_heavy_atoms
                real_pred_energies = real_pred_energies / batch.num_heavy_atoms

                pred_energies = torch.cat(
                    [fake_pred_energies[idxs_to_compute_energies],
                     real_pred_energies[idxs_to_compute_energies]], dim=-1)

                fake_true_energies = torch.FloatTensor(fake_true_energies).to(
                    pred_energies.device)
                real_true_energies = torch.FloatTensor(real_true_energies).to(
                    pred_energies.device)

                pred_energies_delta = (fake_pred_energies - real_pred_energies)[
                    idxs_to_compute_energies]
                true_energies_delta = fake_true_energies - real_true_energies
                energy_loss = nn.L1Loss()(pred_energies_delta,
                                          true_energies_delta)

                mean_red = true_energies_delta.mean()

                log_dict['wgan_D_energy_loss'] = energy_loss
                log_dict['stats_red_metric'] = mean_red

                if not torch.isnan(energy_loss).any():
                    loss += self.wgan_energy_loss_coeff * energy_loss

            log_dict['loss_D'] = loss

        return loss, log_dict

    def configure_optimizers(self):
        optimizers = []

        # generator/encoder optimizers
        param_g = []
        param_g.extend(self.generator.parameters())
        if self.ae_part_type != 'none':
            param_g.extend(self.encoder.parameters())
        opt_g = torch.optim.Adam(param_g,
                                 lr=1e-4, betas=(0.9, 0.999))
        optimizers.append(opt_g)

        # discriminator optimizers
        param_d = []
        if self.ae_part_type == 'aae':
            param_d.append(
                {'params': self.aae_discriminator.parameters(), 'lr': 1e-4,
                 'betas': (0.5, 0.999)})
        if self.use_wgan_part:
            param_d.append(
                {'params': self.gan_discriminator.parameters(), 'lr': 1e-4,
                 'betas': (0.9, 0.999)})

        if len(param_d):
            opt_d = torch.optim.Adam(param_d)

            optimizers.append(opt_d)

        return optimizers
