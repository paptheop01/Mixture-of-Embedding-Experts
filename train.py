# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle 
import torch as th
from torch.utils.data import Dataset, DataLoader
import LSMDC as LD2
import MSRVTT as MSR
import numpy as np
import torch.optim as optim
import argparse
from loss import MaxMarginRankingLoss
from model import Net
from torch.autograd import Variable
import os
import random
from qcm_sampler import QCMSampler
from MSR_sampler import MSRSampler

parser = argparse.ArgumentParser(description='LSMDC2017')

parser.add_argument('--coco', type=bool, default=False,
                    help='add coco dataset')

parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--text_cluster_size', type=int, default=32,
                    help='Text cluster size')
parser.add_argument('--margin', type=float, default=0.2,
                    help='MaxMargin margin value')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='Learning rate exp epoch decay')
parser.add_argument('--n_display', type=int, default=100,
                    help='Information display frequence')
parser.add_argument('--GPU', type=bool, default=True,
                    help='Use of GPU')
parser.add_argument('--n_cpu', type=int, default=1,
                    help='Number of CPU')

parser.add_argument('--model_name', type=str, default='test',
                    help='Model name')
parser.add_argument('--seed', type=int, default=1,
                    help='Initial Random Seed')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Nesterov Momentum for SGD')

parser.add_argument('--eval_qcm', type=bool, default=False,
                    help='Eval or not QCM')

parser.add_argument('--MSRVTT', type=bool, default=False,
                    help='MSRVTT')

parser.add_argument('--coco_sampling_rate', type=float, default=1.0,
                    help='coco sampling rate')

args = parser.parse_args()

print(args)

root_feat = os.path.join('/home/paptheop01/drive/data', 'data')

mp_visual_path = os.path.join(root_feat, 'X_resnet.npy')
mp_flow_path = os.path.join(root_feat, 'X_flow.npy')
mp_face_path = os.path.join(root_feat, 'X_face.npy')


def verbose(epoch, status, metrics, name='TEST'):
    print(name + ' - epoch: %d, epoch status: %.2f, r@1: %.3f, r@5: %.3f, r@10: %.3f, mr: %d' %
          (epoch + 1, status,
           metrics['R1'], metrics['R5'], metrics['R10'],
           metrics['MR']))


def compute_metric(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1

    return metrics


def make_tensor(l, max_len):
    tensor = np.zeros((len(l), max_len, l[0].shape[-1]))
    for i in range(len(l)):
        if len(l[i]):
            tensor[i, :min(max_len, l[i].shape[0]), :] = l[i][:min(max_len, l[i].shape[0])]

    return th.from_numpy(tensor).float()


# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.eval_qcm and not (args.MSRVTT):
    qcm_dataset = LD2.LSMDC_qcm(os.path.join(root_feat, 'resnet-qcm.npy'),
                                os.path.join(root_feat, 'w2v_LSMDC_qcm.npy'),
                                os.path.join(root_feat, 'X_audio_test.npy'),
                                os.path.join(root_feat, 'flow-qcm.npy'),
                                os.path.join(root_feat, 'face-qcm.npy'))

    qcm_sampler = QCMSampler(len(qcm_dataset))
    qcm_dataloader = DataLoader(qcm_dataset, batch_size=500, sampler=qcm_sampler, num_workers=1)
    qcm_gt_fn = os.path.join(root_feat, 'multiple_choice_gt.txt')
    qcm_gt = [line.rstrip('\n') for line in open(qcm_gt_fn)]
    qcm_gt = np.array(map(int, qcm_gt))

print('Pre-loading features ... This may takes several minutes ...')

if args.MSRVTT:
    visual_feat_path = os.path.join(root_feat, 'resnet_features.pickle')
    flow_feat_path = os.path.join(root_feat, 'flow_features.pickle')
    text_feat_path = os.path.join(root_feat, 'w2v_MSRVTT.pickle')
    audio_feat_path = os.path.join(root_feat, 'audio_features.pickle')
    face_feat_path = os.path.join(root_feat, 'face_features.pickle')
    train_list_path = os.path.join(root_feat, 'train_list.txt')
    test_list_path = os.path.join(root_feat, 'test_list.txt')

    dataset = MSR.MSRVTT(visual_feat_path, flow_feat_path, text_feat_path,
                         audio_feat_path, face_feat_path, train_list_path, test_list_path, coco=args.coco)
    msr_sampler = MSRSampler(dataset.n_MSR, dataset.n_coco, args.coco_sampling_rate)

    if args.coco:
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=msr_sampler, num_workers=1, collate_fn=dataset.collate_data, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=1, collate_fn=dataset.collate_data, drop_last=True)

else:
    path_to_text = os.path.join(root_feat, 'w2v_LSMDC.npy')
    path_to_audio = os.path.join(root_feat, 'X_audio_train.npy')

    dataset = LD2.LSMDC(mp_visual_path, path_to_text,
                        path_to_audio, mp_flow_path, mp_face_path, coco=args.coco)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=1, drop_last=True)
    print('Done.')

    print('Reading test data ...')
    resnet_features_path = os.path.join(root_feat, 'resnet152-retrieval.npy.tensor.npy')
    flow_features_path = os.path.join(root_feat, 'flow-retrieval.npy.tensor.npy')
    face_features_path = os.path.join(root_feat, 'face-retrieval.npy.tensor.npy')
    text_features_path = os.path.join(root_feat, 'w2v_LSMDC_retrieval.npy')
    audio_features_path = os.path.join(root_feat, 'X_audio_retrieval.npy.tensor.npy')

    vid_retrieval = np.load(resnet_features_path)
    flow_retrieval = np.load(flow_features_path)
    face_retrieval = np.load(face_features_path)
    text_retrieval = np.load(text_features_path, encoding='latin1')
    audio_retrieval = np.load(audio_features_path)

    mm = max(map(len, text_retrieval))

    text_retrieval = make_tensor(text_retrieval, mm)

    vid_retrieval = th.from_numpy(vid_retrieval).float()
    flow_retrieval = th.from_numpy(flow_retrieval).float()
    face_retrieval = th.from_numpy(face_retrieval).float()
    audio_retrieval = th.from_numpy(audio_retrieval).float()

    text_retrieval_val = text_retrieval
    vid_retrieval_val = vid_retrieval
    flow_retrieval_val = flow_retrieval
    face_retrieval_val = face_retrieval
    audio_retrieval_val = audio_retrieval

    face_ind_test = np.load(os.path.join(root_feat, 'no_face_ind_retrieval.npy'))
    face_ind_test = 1 - face_ind_test
print('Done.')
with open('/mnt/disks/disk-1/train_loader_ant_test.pkl', 'wb') as file:
    pickle.dump(dataloader, file)



