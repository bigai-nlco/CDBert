import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version
from tqdm import tqdm
import numpy as np
import wandb
import logging
from pprint import pprint


import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast

from transformers import BertModel

from utils.param import parse_args
from utils.utils import LossMeter, load_state_dict
from utils.dist_utils import reduce_dict
from models.trainer_base import TrainerBase
from models.dict_modeling_bert import BertClipModel
from dict_pretrain.dict_pretrain_data import get_loader
from dict_pretrain.dict_pretrain_model import BertCLPretraining, BertEGPretraining


_use_native_amp = False
_use_apex = False
_use_native_amp = True


class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        model_kwargs = {}
        # if 'bert' in args.backbone:
        #     model_class = BertCLPretraining
        # print(args)
        config = self.create_config()
        print(config.eg, config.cl)
        # self.model = BertCLPretraining(config)
        self.model = BertEGPretraining(config)
        # self.model = BertEGPretraining(config)
        self.tokenizer = self.create_tokenizer()
        self.model.bert = self.create_model(BertClipModel, config, **model_kwargs)
        # self.model.oribert = self.create_model(BertModel, config, **model_kwargs)
        if 'bert' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            # self.model.oribert = self.load_model(BertModel, config, **model_kwargs)
            self.load_checkpoint(args.load+'.pth')
        else:
            self.model.oribert = self.create_model(BertModel, config, **model_kwargs)
            # ckpt_path = args.load + '.pth'
            # self.load_checkpoint(ckpt_path)
            # # self.start_epoch = int(args.load.split('Epoch')[-1])

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')
    
    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("vis_encoder."):
                new_key = 'encoder.' + key[len("vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("model.vis_encoder."):
                new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)
                
            # if key.startswith('bert'):
            #     new_key = key[len('bert.'):]
            #     state_dict[new_key] = state_dict.pop(key)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)
            
    def load_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.load

        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model
    
    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 9595.


            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            # wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=80)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):
                # continue
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            # desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                            desc_str += f' {loss_name}  {loss_meter.val:.2f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()
            if self.args.distributed:
                dist.barrier()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n" + f""

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                            # wandb.log({f'Train Loss/{name}': avg_loss}, step=epoch)

                losses_str += '\n'
                print(losses_str)
            if self.args.distributed:
                dist.barrier()

            # Validation
            valid_results, valid_uid2ans = self.evaluate_epoch(epoch=epoch)

            valid_results = reduce_dict(valid_results, average=False)
            if self.verbose:
                valid_loss = valid_results['total_loss']
                valid_loss_count = valid_results['total_loss_count']

                avg_valid_loss = valid_loss / valid_loss_count
                losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                for name, loss in valid_results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(valid_results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                            # wandb.log({f'Valid Loss/{name}': avg_loss}, step=epoch)

                losses_str += '\n'
                print(losses_str)
            
            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                # Save
                if avg_valid_loss < best_eval_loss:
                    best_eval_loss = avg_valid_loss
                #     self.save("BEST_EVAL_LOSS")
                self.save("Epoch%02d" % (epoch + 1))

            if self.args.distributed:
                dist.barrier()

        # if self.verbose:
        #     wandb.log({'finished': True})

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        uid2ans = {}

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=80)

            for step_i, batch in enumerate(self.val_loader):

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                if 'qa' in self.args.losses:
                    qa_pred = results['qa_pred']
                    for uid, ans in zip(batch['uid'], qa_pred):
                        uid2ans[uid] = ans

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                pbar.close()
            if self.args.distributed:
                dist.barrier()

            if 'qa' not in self.args.losses:
                uid2ans = None

            return epoch_results, uid2ans

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        data_path=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
        cl=True)

    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        data_path=args.valid, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.valid_topk,
        cl=True)

    trainer = Trainer(args, train_loader, val_loader, train=True)

    trainer.train()

def debug_worker(args):
    args.gpu = 0
    train_loader = get_loader(
        args,
        data_path=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
        cl=True)

    val_loader = get_loader(
        args,
        data_path=args.valid, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.valid_topk,
        cl=True)

    trainer = Trainer(args, train_loader, val_loader, train=True)

    trainer.train()    

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss') # total loss
    LOSSES_NAME.append('cl_loss')
    LOSSES_NAME.append('eg_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'coco' in args.train:
        dsets.append('COCO')
    if 'vg' in args.train:
        dsets.append('VG')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
    else:
        debug_worker(args)
