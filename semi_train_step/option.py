import argparse
from reid import datasets
import os.path as osp

parser = argparse.ArgumentParser(description='MGN')

parser.add_argument('--nThread', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')

parser.add_argument("--datadir", type=str, default="Market-1501-v15.09.15", help='dataset directory')
parser.add_argument('--data_train', type=str, default='Market1501', help='train dataset name')
parser.add_argument('--data_test', type=str, default='Market1501', help='test dataset name')

parser.add_argument('--reset', action='store_true', help='reset the training')
parser.add_argument('--test_every', type=int, default=20, help='do test per every N epochs')
parser.add_argument("--batchid", type=int, default=16, help='the batch for id')
parser.add_argument("--batchimage", type=int, default=4, help='the batch of per id')
parser.add_argument("--batchtest", type=int, default=32, help='input batch size for test')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

parser.add_argument('--model', default='MCC', help='model name')
parser.add_argument('--loss', type=str, default='1*CrossEntropy+1*Triplet', help='loss function configuration')

parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--pool', type=str, default='avg', help='pool function')
parser.add_argument('--feats', type=int, default=256, help='number of feature maps')
parser.add_argument('--num_classes', type=int, default=3131, help='')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')


parser.add_argument('--optimizer', default='ADAM', choices=('SGD','ADAM','NADAM','RMSprop'), help='optimizer to use (SGD | ADAM | NADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--dampening', type=float, default=0, help='SGD dampening')
parser.add_argument('--nesterov', action='store_true', help='SGD nesterov')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--amsgrad', action='store_true', help='ADAM amsgrad')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor for step decay')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--lr_decay', type=int, default=60, help='learning rate decay per N epochs')

parser.add_argument("--re_rank", action='store_true', help='')
parser.add_argument("--random_erasing", action='store_true', help='')
parser.add_argument("--probability", type=float, default=0.5, help='')

parser.add_argument("--savedir", type=str, default='saved_models', help='directory name to save')
parser.add_argument("--outdir", type=str, default='out', help='')
#parser.add_argument("--resume", type=int, default=0, help='resume from specific checkpoint')
#parser.add_argument("--resume", type=int, default=-1, help='resume from specific checkpoint')
parser.add_argument('--save', type=str, default='test', help='file name to save')
parser.add_argument('--load_dir', type=str, default='/home/h/MyWorks/SSGP_aichanllenge/logs/model_280.pt', help='file name to load')
parser.add_argument('--save_models', action='store_true', help='save all intermediate models')
parser.add_argument('--pre_train', type=str, default='PED_EXT_008.pt', help='pre-trained model directory')


# data
parser.add_argument('--src-dataset', type=str, default='market1501',
                    choices=datasets.names())
parser.add_argument('--tgt-dataset', type=str, default='dukemtmc',  #dukemtmc',
                    choices=datasets.names())
#parser.add_argument('--ikea-dataset', type=str, default='ikea')            #ikea data
parser.add_argument('-b', '--batch-size', type=int, default=64)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--split', type=int, default=0)

parser.add_argument('--combine-trainval', action='store_true',
                    help="train and val sets together for training, "
                            "val set alone for validation")
parser.add_argument('--num_instances', type=int, default=4,
                    help="each minibatch consist of "
                            "(batch_size // num_instances) identities, and "
                            "each identity has num_instances instances, "
                            "default: 4")
# model
parser.add_argument('--num-features', type=int, default=2048, help='')
""" parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    choices=models.names()) """
parser.add_argument('--features', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0)
# loss
parser.add_argument('--margin', type=float, default=0.5,
                    help="margin of the triplet loss, default: 0.5")
parser.add_argument('--lambda_value', type=float, default=0.1,
                    help="balancing parameter, default: 0.1")
parser.add_argument('--rho', type=float, default=1.6e-3,
                    help="rho percentage, default: 1.6e-3")
parser.add_argument('--mode', type=str,  default="Dissimilarity",
                    choices=["Classification", "Dissimilarity", "Weight"])
# optimizer
parser.add_argument('--lr', type=float, default=6e-5,
                    help="learning rate of all parameters")
parser.add_argument('--weight-decay', type=float, default=5e-4)
# training configs
parser.add_argument('--resume', type=str, metavar='PATH',
                    default='/home/h/MyWorks/SSGP_aichanllenge/logs/model_best.pth.tar')
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--start-epoch', type=int, default=0,
                    help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=20)
parser.add_argument('--num_split', type=int, default=1)
parser.add_argument('--iteration', type=int, default=25)
parser.add_argument('--no-rerank', action='store_true', help="train without rerank")
parser.add_argument('--dce-loss', action='store_true', help="train without rerank")
parser.add_argument('--sample', type=str, default='cluster', choices=['random', 'cluster'])
# metric learning
parser.add_argument('--dist-metric', type=str, default='euclidean',
                    choices=['euclidean', 'kissme'])
# misc
working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--data-dir', type=str, metavar='PATH',
                    default='/home/h/MyWorks/SSGP_aichanllenge/dataset')
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs/ai_challenge'))
parser.add_argument('--load-dist', action='store_true', help='load pre-compute distance')
parser.add_argument('--gpu-devices', default='0', type=str, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('-a', '--arch', type=str, default='resnet50')



args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

