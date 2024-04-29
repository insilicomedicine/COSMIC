import numpy as np

import torch
from torch_geometric.data import DataLoader, Batch

import sys

sys.path.append('.')

from energy_minimization.data import ConfDataset
from energy_minimization.utils.utils import (add_prop, compute_alligment,
                                             get_energy, set_conformer,
                                             compute_desciptors3d)

from energy_minimization.models.cosmic import COSMIC

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')

import argparse


class ConfGenModel(pl.LightningModule):
    def __init__(self,
                 model='aae_wgan',
                 conditions='none',
                 dataset='drugs',
                 num_samples=10,
                 
                 # params for ablation studies
                 num_ref_steps=10,
                 lambda_cosloss=0.5,
                 lambda_mxloss=1.0,
                 lambda_energyloss=0.1):
        super().__init__()

        self.num_samples = num_samples

        models_dict = {'aae': {'use_wgan_part': False, 'ae_part_type': 'aae'},
                       'vae': {'use_wgan_part': False, 'ae_part_type': 'vae'},
                       'wgan': {'use_wgan_part': True, 'ae_part_type': 'none'},
                       'aae_wgan': {'use_wgan_part': True,
                                    'ae_part_type': 'aae'},
                       'vae_wgan': {'use_wgan_part': True,
                                    'ae_part_type': 'vae'}
                       }
        model_params_by_dataset = {
            'drugs': {'latent_size': 3,

                      'node_hidden_size': 128,
                      'edge_hidden_size': 64,

                      'num_gaussians': 64,

                      'num_backbone_layers': 4,
                      'num_main_layers': 6,
                      'num_refiner_steps': num_ref_steps,
                      'lambda_cosloss': lambda_cosloss,
                      'lambda_mxloss': lambda_mxloss,
                      'ae_num_encoder_layers': 4,

                      'vae_kl_beta': 0.03,

                      'aae_num_discriminator_layers': 4,
                      'aae_discr_coeff': 0.01,

                      'wgan_num_discr_inter': 6,
                      'wgan_lambda_gp': 10.,
                      'wgan_discr_coeff': 0.01,
                      'wgan_num_mols_energy': 32,
                      'wgan_energy_loss_coeff': lambda_energyloss,

                      'num_warmup_iteration': 400
                      },
            'qm9': {'latent_size': 3,

                    'node_hidden_size': 128,
                    'edge_hidden_size': 64,

                    'num_gaussians': 64,

                    'num_backbone_layers': 4,
                    'num_main_layers': 6,
                    'num_refiner_steps': num_ref_steps,
                    'lambda_cosloss': lambda_cosloss,
                    'lambda_mxloss': lambda_mxloss,
                    'ae_num_encoder_layers': 4,

                    'vae_kl_beta': 0.01,

                    'aae_num_discriminator_layers': 4,
                    'aae_discr_coeff': 0.01,

                    'wgan_num_discr_inter': 6,
                    'wgan_lambda_gp': 10.,
                    'wgan_discr_coeff': 0.01,
                    'wgan_num_mols_energy': 32,
                    'wgan_energy_loss_coeff': lambda_energyloss,

                    'num_warmup_iteration': 200
                    }
        }

        model_params = model_params_by_dataset[dataset]
        model_params.update(models_dict[model])
        model_params.update({'conditions': conditions})

        self.model = COSMIC(**model_params)

    def forward(self, batch):
        nodes_coords_preds, torsion_preds = self.model(batch)

        return nodes_coords_preds, torsion_preds

    def training_step(self, batch, batch_idx, optimizer_idx: int = 0):
        loss, log_dict = self.model.compute_loss(batch, batch_idx,
                                                 self.current_epoch,
                                                 optimizer_idx)
        for k in log_dict.keys():
            self.log('train_' + k, log_dict[k],
                     on_step=True,
                     prog_bar=True)
        loss.masked_fill_(torch.isnan(loss), 0.0)
        return loss

    def validation_step(self, batch: Batch, batch_nb):
        true_confs = [d.cartesian_y
                      for d in batch.to_data_list()]

        sampled_confs = [[] for _ in range(len(true_confs))]

        for _ in range(self.num_samples):
            nodes_out = self.model.sample(batch)

            add_prop(batch, 'cartesian_pred', nodes_out)

            for i, d in enumerate(batch.to_data_list()):
                sampled_confs[i].append(d.cartesian_pred)

        mols = batch.mol

        return true_confs, sampled_confs, mols

    def validation_epoch_end(self, outputs):
        true_3d_pointclouds = sum([a[0] for a in outputs], [])
        sampled_3d_pointclouds = sum([a[1] for a in outputs], [])
        mols = sum([a[2] for a in outputs], [])

        true_mols = []
        sampled_mols = []

        icrmse = []

        for i, (tr, sampled_list, mol) in enumerate(
                zip(true_3d_pointclouds,
                    sampled_3d_pointclouds,
                    mols)):
            tr = tr - torch.mean(tr, dim=0, keepdim=True)
            true_mols.append(set_conformer(mol, tr))
            sampled_mols.append([])

            for sample in sampled_list:
                tr, sample = compute_alligment(tr, sample)
                sampled_mols[-1].append(set_conformer(mol, sample))

            cur_rmse_list = []
            for i in range(len(sampled_list)):
                for j in range(i + 1, len(sampled_list)):
                    sample_i = sampled_list[i]
                    sample_j = sampled_list[j]

                    sample_i, sample_j = compute_alligment(sample_i, sample_j)

                    cur_rmse_list.append(
                        torch.sqrt(
                            ((sample_i - sample_j) ** 2).sum(dim=-1).mean())
                    )

            icrmse.append(torch.stack(cur_rmse_list).mean())

        energies = [([get_energy(sample_mol, normalize=True, addHs=True) for
                      sample_mol in sample_mol_list],
                     get_energy(true_mol, normalize=True, addHs=True))
                    for true_mol, sample_mol_list in
                    zip(true_mols, sampled_mols)]

        energies_flattened = [(sample_en, true_en)
                              for sample_en_list, true_en in energies
                              for sample_en in sample_en_list]

        energy_diff = [(sample_en - true_en)
                       for sample_en_list, true_en in energies
                       for sample_en in sample_en_list
                       if not np.isnan(sample_en - true_en)]

        quantiles = np.quantile(energy_diff,
                                [0.5, 0.25, 0.75, 0.99])

        neg_diff_proc = np.mean([diff < 0. for diff in energy_diff])

        descriptors3d = \
            [([compute_desciptors3d(sample_mol)
               for sample_mol in sample_mol_list],
              compute_desciptors3d(true_mol))
             for true_mol, sample_mol_list in zip(true_mols, sampled_mols)]

        descriptors3d_flattened = [(sample_descr, true_descr)
                                   for sample_descrs, true_descr in
                                   descriptors3d
                                   for sample_descr in sample_descrs]

        descriptors3d_pred = [t[0].numpy() for t in descriptors3d_flattened]
        descriptors3d_true = [t[1].numpy() for t in descriptors3d_flattened]

        descriptors3d_pred = np.array(descriptors3d_pred)
        descriptors3d_true = np.array(descriptors3d_true)

        descriptors_mean_corrcoef = \
            np.mean([np.corrcoef(descriptors3d_true[:, i],
                                 descriptors3d_pred[:, i])[0, 1]
                     for i in range(descriptors3d_true.shape[1])])

        if self.logger is not None:
            combined_mols = []
            for true_mol, sample_mol_list, (sample_en_list, true_en) \
                    in zip(true_mols, sampled_mols, energies):
                combined_mol = true_mol
                for sample_mol in sample_mol_list:
                    combined_mol = Chem.CombineMols(combined_mol, sample_mol)

                en_median = np.quantile([(sample_en - true_en)
                                         for sample_en in sample_en_list],
                                        0.5)
                combined_mols.append((combined_mol, en_median))

            combined_mols_with_en = np.random.permutation(combined_mols)
            sorted_combined_mols = sorted(combined_mols_with_en,
                                          key=lambda x: x[-1])

            try:
                self.draw_mols([p[0] for p in combined_mols[:64]],
                               ['{:.7}'.format(p[1]) for p in
                                combined_mols[:64]],
                               'samples_random', 8,
                               self.logger.experiment,
                               self.trainer.global_step)
                self.draw_mols([p[0] for p in sorted_combined_mols[:64]],
                               ['{:.7}'.format(p[1]) for p in
                                sorted_combined_mols[:64]],
                               'samples_min_diff', 8,
                               self.logger.experiment,
                               self.trainer.global_step)
                self.draw_mols([p[0] for p in sorted_combined_mols[-64:]],
                               ['{:.7}'.format(p[1]) for p in
                                sorted_combined_mols[-64:]],
                               'samples_max_diff', 8,
                               self.logger.experiment,
                               self.trainer.global_step)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 1, 1)

                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)

                ax.set_xlabel('True energy')
                ax.set_ylabel('Energy of generated object')

                ax.scatter([e[1] for e in energies_flattened],
                           [e[0] for e in energies_flattened],
                           s=2)

                ax.plot([-10, 10],
                        [-10, 10], c='violet')
                ax.plot([-10, 10],
                        [-9, 11], c='violet', label='diff=100')

                median = quantiles[0]
                ax.plot([-10, 10],
                        [-10 + median, 10 + median], c='b',
                        label='median')

                q1 = quantiles[1]
                ax.plot([-10, 10],
                        [-10 + q1, 10 + q1], c='g',
                        label='q1')

                q3 = quantiles[2]
                ax.plot([-10, 10],
                        [-10 + q3, 10 + q3], c='r',
                        label='q3')

                ax.legend()

                self.logger.experiment.add_figure('energies', fig,
                                                  global_step=self.trainer.global_step)
                self.logger.experiment.flush()
            except:
                pass

        icrmse = torch.stack(icrmse).mean()

        log_dict = {
            'energy_diff_median': quantiles[0],
            'energy_diff_q1': quantiles[1],
            'energy_diff_q3': quantiles[2],
            'energy_diff_p99': quantiles[3],
            'neg_diff_proc': neg_diff_proc,
            'icrmse': icrmse,
            'descr_corr': descriptors_mean_corrcoef
        }

        for k in log_dict.keys():
            self.log('val_' + k, log_dict[k], on_epoch=True)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    @staticmethod
    def draw_mols(mols, legends, name, molsPerRow, experiment, step):
        experiment.add_image(name,
                             np.array(
                                 Draw.MolsToGridImage(mols,
                                                      molsPerRow=molsPerRow,
                                                      legends=legends)
                             ) / 255,
                             global_step=step,
                             dataformats='HWC')
        experiment.flush()


def train_conf_gen(gpus,
                   root,
                   summary_path,
                   pretrained_weights_path=None,
                   model='predictor',
                   dataset='drugs',
                   task_type='argmin',
                   batch_size=32,
                   conditions='none',
                   perform_logging=True,
                   save_checkpoints=True,
                   num_epochs=1,
                   num_workers=1,
                   num_ref_steps=10,
                   lambda_cosloss=0.5,
                   lambda_mxloss=1.0,
                   lambda_energyloss=0.1,
                   verbose=True,
                   fast_dev_run=False,
                   seed=777):
    experiment_name = f'conf_generator_{model}_{num_ref_steps}refsteps_{conditions}_{dataset}_{seed}'

    if lambda_cosloss == 0.0:
        experiment_name += '_no_cosloss'
    
    if lambda_mxloss == 0.0:
        experiment_name += '_no_mxloss'
        
    if lambda_energyloss == 0.0:
        experiment_name += '_no_energyloss'
    
    if pretrained_weights_path is not None:
        experiment_name += '_from_pretrained'

    if perform_logging:
        logger = loggers.TensorBoardLogger(save_dir='./logs',
                                           name=experiment_name,
                                           max_queue=1)
    else:
        logger = None

    callbacks = []
    if save_checkpoints:
        callbacks.append(ModelCheckpoint(
            dirpath='./saved_models/' + experiment_name,
            save_weights_only=True))

    gen_model = ConfGenModel(model=model, conditions=conditions,
                             dataset=dataset, 
                             num_ref_steps=num_ref_steps,
                             lambda_cosloss=lambda_cosloss,
                             lambda_mxloss=lambda_mxloss)
    if pretrained_weights_path is not None:
        weights = torch.load(pretrained_weights_path,
                             map_location='cpu')['state_dict']
        gen_model.load_state_dict(state_dict=weights, strict=False)

    train_dl = DataLoader(
        ConfDataset(root,
                    summary_path,
                    split='train',
                    conditions=conditions,
                    task_type=task_type),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)

    val_dl = DataLoader(
        ConfDataset(root,
                    summary_path,
                    split='val',
                    conditions=conditions,
                    task_type='argmin'),
        batch_size=batch_size, num_workers=num_workers)

    val_check_interval_dict = {'drugs': 0.1,
                               'qm9': 1.0}

    tr = pl.Trainer(
        gpus=gpus,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=1,
        val_check_interval=val_check_interval_dict[dataset] if verbose else 0.0,
        gradient_clip_val=1.0,
        terminate_on_nan=True,
        weights_save_path=f'../saved_models/{experiment_name}'
        if save_checkpoints else None,
        precision=32 if (gpus is None) else 16,
        fast_dev_run=fast_dev_run)

    tr.fit(gen_model, train_dataloader=train_dl, val_dataloaders=val_dl)

    return gen_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='aae_wgan')
    parser.add_argument('--dataset', type=str, default='drugs')

    parser.add_argument('--gpu', action='append')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=1)

    parser.add_argument('--num_ref_steps', type=int, default=10)
    parser.add_argument('--lambda_cosloss', type=float, default=0.5)
    parser.add_argument('--lambda_mxloss', type=float, default=1.0)
    parser.add_argument('--lambda_energyloss', type=float, default=0.1)

    parser.add_argument('--conditions', type=str, default='none')
    parser.add_argument('--task_type', type=str, default='distr_learn')

    parser.add_argument('--pretrained_weights_path', type=str, default='none')
    
    parser.add_argument('--root', type=str)
    parser.add_argument('--summary_path', type=str)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    train_conf_gen(
        model=args.model,
        gpus=None if (args.gpu is None) else [int(gpu_id) for gpu_id in
                                              args.gpu],
        pretrained_weights_path=None if args.pretrained_weights_path == 'none' else args.pretrained_weights_path,
        root=args.root,
        summary_path=args.summary_path,
        batch_size=args.batch_size,
        num_ref_steps=args.num_ref_steps,
        lambda_cosloss=args.lambda_cosloss,
        lambda_mxloss=args.lambda_mxloss,
        lambda_energyloss=args.lambda_energyloss,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        conditions=args.conditions,
        task_type=args.task_type,
        dataset=args.dataset,
        seed=args.seed
    )
