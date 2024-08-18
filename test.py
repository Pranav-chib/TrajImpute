import argparse
import numpy as np
import torch
import os
import time
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from dataset import TrajectoryDataset
from model import TrajectoryModel

warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--dataset_name', type=str, default='zara1')
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Function to visualize predicted trajectories
def vis_predicted_trajectories(obs_traj, gt, pred_trajs, pred_probabilities, min_index):
    background_path = 'zara.PNG'
    background_img = mpimg.imread(background_path)
    rotated_image = ndimage.rotate(background_img, 90)

    for i in range(obs_traj.shape[0]):
        plt.clf()
        curr_obs = obs_traj[i].cpu().numpy()  # [T_obs 2]
        curr_gt = gt[i].cpu().numpy()
        curr_preds = pred_trajs[i].cpu().numpy()
        curr_pros = pred_probabilities[i].cpu().numpy()
        curr_min_index = min_index[i].cpu().numpy()

        obs_x = curr_obs[:, 0]
        obs_y = curr_obs[:, 1]
        gt_x = np.concatenate((obs_x[-1:], curr_gt[:, 0]))
        gt_y = np.concatenate((obs_y[-1:], curr_gt[:, 1]))

        plt.plot(obs_x, obs_y, color='tab:blue', linestyle='-', label='Observed', markersize=7)
        plt.plot(gt_x, gt_y, color='tab:green', linestyle='-', label='Ground Truth', markersize=7)
        plt.imshow(background_img, extent=[-7, 3, -5, 5])

        for j in range(curr_preds.shape[0]):
            pred_x = np.concatenate((obs_x[-1:], curr_preds[j][:, 0]))
            pred_y = np.concatenate((obs_y[-1:], curr_preds[j][:, 1]))

            if j == curr_min_index:
                plt.plot(pred_x, pred_y, ls='--', lw=2.0, color='tab:orange', label='Predicted', markersize=7)
            else:
                plt.plot(pred_x, pred_y, ls='--', lw=0.5, color='tab:red', markersize=5)

        plt.tight_layout()
        save_path = './fig/' + args.dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + str(time.time()) + '.png')

# Function to test the model
def test(model, test_dataloader):
    model.eval()
    ade = 0
    fde = 0
    num_traj = 0

    for (ped, neis, mask) in test_dataloader:
        ped = ped.cuda()
        neis = neis.cuda()
        mask = mask.cuda()

        ped_obs = ped[:, :args.obs_len]
        gt = ped[:, args.obs_len:]
        neis_obs = neis[:, :, :args.obs_len]

        with torch.no_grad():
            num_traj += ped_obs.shape[0]

            pred_trajs, scores = model(ped_obs, neis_obs, mask, None, test=True)
            top_k_scores = torch.topk(scores, k=20, dim=-1).values
            top_k_scores = F.softmax(top_k_scores, dim=-1)
            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
            gt_ = gt.unsqueeze(1)
            norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
            ade_ = torch.mean(norm_, dim=-1)
            fde_ = norm_[:, :, -1]
            min_ade, min_ade_index = torch.min(ade_, dim=-1)
            min_fde, min_fde_index = torch.min(fde_, dim=-1)

            vis_predicted_trajectories(ped_obs, gt, pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[-2], -1), top_k_scores, min_fde_index)

            ade += torch.sum(min_ade).item()
            fde += torch.sum(min_fde).item()

    ade = ade / num_traj
    fde = fde / num_traj
    return ade, fde, num_traj

# Load the test dataset and model
test_dataset = TrajectoryDataset(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                 dataset_type='test', translation=True, rotation=True, scaling=False,
                                 obs_len=args.obs_len)

test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn, batch_size=128, shuffle=False)

model = TrajectoryModel(in_size=2, obs_len=args.obs_len, pred_len=args.pred_len, embed_size=64,
                        enc_num_layers=1, int_num_layers_list=[1, 1], heads=4, forward_expansion=2)
model = model.cuda()

# Load the saved model checkpoint
model.load_state_dict(torch.load(os.path.join(args.checkpoint, args.dataset_name, 'best.pth')))
model.eval()

# Test the model
ade, fde, num_traj = test(model, test_loader)

print('Test Results:')
print('ADE:', ade)
print('FDE:', fde)
print('Number of Trajectories:', num_traj)
