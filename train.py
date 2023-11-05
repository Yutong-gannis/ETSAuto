import os
import time
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from opt import get_opts
from dataloader import ETSMotion
from model import PlanModel
from loss import MultipleTrajectoryPredictionLoss


def get_dataloader(dataset_path, batch_size, num_workers, split=0.8):
    etsmotion = ETSMotion(dataset_path)
    train_size = int(len(etsmotion) * split)
    val_size = len(etsmotion) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(etsmotion, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, val_loader


def configure_optimizers(args, model):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, )
    else:
        raise NotImplementedError
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
    return optimizer, lr_scheduler


def main(args):
    writer = SummaryWriter()
    train_dataloader, val_dataloader = get_dataloader(args.dataset, args.batch_size, args.num_workers, split=0.5)
    model = PlanModel(num_cls=args.M, num_pts=args.num_pts).cuda()
    if args.accuracy == 'half':
        model = model.half()
        scaler = GradScaler()
    elif args.accuracy == 'mix':
        scaler = GradScaler()
        
    optimizer, lr_scheduler = configure_optimizers(args, model)
    if args.resume:
        print('Loading weights from', args.resume)
        model.load_state_dict(torch.load(args.resume), strict=True)
        
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')
    #loss = nn.SmoothL1Loss(reduction='none')

    num_steps = 0
    disable_tqdm = not args.tqdm

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):
        for data in tqdm(train_dataloader, leave=False, disable=disable_tqdm, position=1):
            front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = data
            #front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = front_sequence.cuda(), leftrear_sequence.cuda(), rightrear_sequence.cuda(), nav_sequence.cuda(), hist_trajectory_sequence.cuda(), speed_limit_sequence.cuda(), stop_sequence.cuda(), traffic_convention_sequence.cuda(), trajectory_sequence.cuda()
            bs = front_sequence.size(0)
            seq_length = front_sequence.size(1)
            
            hist_feature = torch.zeros((bs, 40, 128)).cuda()
            if args.accuracy == 'half':
                hist_feature = hist_feature.half()
            
            total_loss = 0
            for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                num_steps += 1
                front, leftrear, rightrear, nav = front_sequence[:, t, :, :, :], leftrear_sequence[:, t, :, :, :], rightrear_sequence[:, t, :, :, :], nav_sequence[:, t, :, :, :]
                hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = hist_trajectory_sequence[:, t, :, :], speed_limit_sequence[:, t:t+1], stop_sequence[:, t, :], traffic_convention_sequence[:, t, :], trajectory_sequence[:, t, :, :]
                front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.cuda(), leftrear.cuda(), rightrear.cuda(), nav.cuda(), hist_trajectory.cuda(), speed_limit.cuda(), stop.cuda(), traffic_convention.cuda(), trajectory_label.cuda()
                if args.accuracy == 'half':
                    front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.half(), leftrear.half(), rightrear.half(), nav.half(), hist_trajectory.half(), speed_limit.half(), stop.half(), traffic_convention.half(), trajectory_label.half()
                else:
                    front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.float(), leftrear.float(), rightrear.float(), nav.float(), hist_trajectory.float(), speed_limit.float(), stop.float(), traffic_convention.float(), trajectory_label.float()
                with autocast():
                    pred_cls, pred_trajectory, feature = model(front, leftrear, rightrear, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention)
                    feature = feature.clone().detach()
                    hist_feature = torch.cat((hist_feature, feature), dim=1)[:, 1:]
                    cls_loss, reg_loss = loss(pred_cls, pred_trajectory, trajectory_label)
                total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / args.optimize_per_n_step
            
                writer.add_scalar('train/epoch', epoch, num_steps)
                writer.add_scalar('loss/cls', cls_loss, num_steps)
                writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
                writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
                writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
                writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)

                if (t + 1) % args.optimize_per_n_step == 0:
                    optimizer.zero_grad()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.accuracy == 'float':
                        total_loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    writer.add_scalar('loss/total', total_loss, num_steps)
                    total_loss = 0

            if not isinstance(total_loss, int):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
                optimizer.step()
                writer.add_scalar('loss/total', total_loss, num_steps)

        lr_scheduler.step()
        if (epoch + 1) % args.val_per_n_epoch == 0:
            ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
            torch.save(model.module.state_dict(), ckpt_path)
            print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            model.eval()
            with torch.no_grad():
                for data in tqdm(val_dataloader, leave=False, disable=disable_tqdm, position=1):
                    front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = data
                    #front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = front_sequence.cuda(), leftrear_sequence.cuda(), rightrear_sequence.cuda(), nav_sequence.cuda(), hist_trajectory_sequence.cuda(), speed_limit_sequence.cuda(), stop_sequence.cuda(), traffic_convention_sequence.cuda(), trajectory_sequence.cuda()

                    bs = front_sequence.size(0)
                    seq_length = front_sequence.size(1)
                    
                    hist_feature = torch.zeros((bs, 40, 128)).cuda()
                    for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                        front, leftrear, rightrear, nav = front_sequence[:, t, :, :], leftrear_sequence[:, t, :, :], rightrear_sequence[:, t, :, :], nav_sequence[:, t, :, :]
                        hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = hist_trajectory_sequence[:, t, :, :], speed_limit_sequence[:, t:t+1], stop_sequence[:, t, :], traffic_convention_sequence[:, t, :], trajectory_sequence[:, t, :, :]
                        front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.cuda(), leftrear.cuda(), rightrear.cuda(), nav.cuda(), hist_trajectory.cuda(), speed_limit.cuda(), stop.cuda(), traffic_convention.cuda(), trajectory_label.cuda()
                        if args.accuracy == 'half':
                            front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.half(), leftrear.half(), rightrear.half(), nav.half(), hist_trajectory.half(), speed_limit.half(), stop.half(), traffic_convention.half(), trajectory_label.half()
                        else:
                            front, leftrear, rightrear, nav, hist_trajectory, speed_limit, stop, traffic_convention, trajectory_label = front.float(), leftrear.float(), rightrear.float(), nav.float(), hist_trajectory.float(), speed_limit.float(), stop.float(), traffic_convention.float(), trajectory_label.float()
                        with autocast():
                            pred_cls, pred_trajectory, feature = model(front, leftrear, rightrear, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention)
                            feature = feature.clone().detach()
                            hist_feature = torch.cat((hist_feature, feature), dim=1)[:, 1:]
                        
                            cls_loss, reg_loss = loss(pred_cls, pred_trajectory, trajectory_label)
                        total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / model.module.optimize_per_n_step
                    
                        writer.add_scalar('val/epoch', epoch, num_steps)
                        writer.add_scalar('loss/cls', cls_loss, num_steps)
                        writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
                        writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
                        writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
                        writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)
                        
            model.train()


if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...')
    args = get_opts()
    main(args=args)