import time
from argparse import ArgumentParser


def get_opts():
    parser = ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="D:\ETSMotion\ETSMotion")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_per_n_step', type=int, default=20)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)

    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=8)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--optimize_per_n_step', type=int, default=40)
    parser.add_argument('--tqdm', type=bool, default=False)
    parser.add_argument('--accuracy', type=str, default='float')

    exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    args = parser.parse_args()
    return args