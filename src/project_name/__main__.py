import os
import warnings
import hydra
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig
from project_name.datasets.ermes_datamodule import ErmesDataModule
from project_name.models.module import Module
from project_name.utils import start_memory_daemon
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from project_name.utils import cleanup
from lightning.pytorch.loggers import WandbLogger


torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", message=".*The 'val_dataloader' does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", message=".*The 'test_dataloader' does not have many workers which may be a bottleneck.*")
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(cfg, jobid):
    wandb_dir = os.path.join('logs', jobid)
    logger = WandbLogger(name=jobid, project='damage', save_dir=wandb_dir)
    logger.experiment.config.update(dict(cfg))
    datamodule = ErmesDataModule(
        cfg.dataset,
        batch_size=cfg.model.batch_size,
        verbose=True
    )
    model = Module(cfg.model, jobid=jobid)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filename='best',
        dirpath=f'logs/{jobid}/checkpoints',
    )
    trainer = Trainer(
        max_epochs=cfg.model.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accumulate_grad_batches=cfg.model.accumulate_batches,
        # num_sanity_val_steps=1,
        # limit_train_batches=1,
        # limit_val_batches=1,
        check_val_every_n_epoch=3,
        default_root_dir=f'logs/{jobid}',
    )
    trainer.fit(model, datamodule=datamodule)
    checkpoint_path = checkpoint_callback.best_model_path
    model = Module.load_from_checkpoint(checkpoint_path, config=cfg['model'])
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    cleanup(cfg)


def test(cfg, jobid):
    wandb_dir = os.path.join('logs', jobid)
    logger = WandbLogger(name=jobid, project='damage', save_dir=wandb_dir)
    logger.experiment.config.update(dict(cfg))
    datamodule = ErmesDataModule(
        cfg.dataset,
        batch_size=cfg.model.batch_size,
        verbose=True
    )
    jobid_to_be_tested = cfg.command.ver_number
    checkpoint_path = f"logs/{jobid_to_be_tested}/checkpoints/best.ckpt"
    model = Module.load_from_checkpoint(checkpoint_path, config=cfg.model, jobid=jobid_to_be_tested)
    trainer = Trainer(logger=logger)
    datamodule = ErmesDataModule(
        cfg['dataset'],
        batch_size=cfg['model']['batch_size'],
        verbose=True
    )
    trainer.test(model=model, datamodule=datamodule)
    
    
def lr_find(cfg, jobid):
    wandb_dir = os.path.join('logs', jobid)
    logger = WandbLogger(name=jobid, project='damage', save_dir=wandb_dir)    
    logger.experiment.config.update(dict(cfg))
    datamodule = ErmesDataModule(
        cfg.dataset,
        batch_size=cfg.model.batch_size,
        verbose=True
    )
    model = Module(cfg.model, jobid=jobid)
    trainer = Trainer(
        logger=logger, 
        default_root_dir=f'logs/{jobid}',
        num_sanity_val_steps=0,  # Disable sanity check
    )
    tuner = Tuner(trainer=trainer)
    lr_finder = tuner.lr_find(model,
                              datamodule=datamodule,
                              min_lr=cfg['command']['min_lr'],
                              max_lr=cfg['command']['max_lr'],
                              early_stop_threshold=cfg['command']['es_threshold'],
                              num_training=cfg['command']['num_training'],
                              )
    fig = lr_finder.plot(suggest=True)
    fig.savefig(f'logs/{jobid}/lr_finder.png')
    print(lr_finder.suggestion())


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    start_memory_daemon()
    job_id = os.environ.get('SLURM_JOB_ID', 'debug')
    if cfg['command']['name'] == 'train':
        train(cfg, job_id)
    if cfg['command']['name'] == 'test':
        test(cfg, job_id)
    if cfg['command']['name'] == 'lr_find':
        lr_find(cfg, job_id)


if __name__ == "__main__":
    main()
