import argparse
import importlib
import logging
import sys
import os
from itertools import product
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Test Script')
    parser.add_argument('--model', type=str, default='DDPCC_geo')
    parser.add_argument('--lossless_model', type=str, default='DDPCC_lossless_coder')
    parser.add_argument('--log_name', type=str, default='aaa')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--ckpt_dir', type=str,
                        default='./ddpcc_ckpts')
    parser.add_argument('--pcgcv2_ckpt_dir', type=str,
                        default='./pcgcv2_ckpts')
    parser.add_argument('--frame_count', type=int, default=100, help='number of frames to be coded')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to store results (in csv format)')
    parser.add_argument('--tmp_dir', type=str, default='./my_tmp', help="directory to store temporary ply files that will be encoded by GPCC")
    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the bitstream of previous frame')
    parser.add_argument('--dataset_dir', type=str, default='/home/yelanggao/Dataset/kitti_odometry/dataset')
    return parser.parse_args()
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, './PCGCv2'))

from dataset_kitti import *
# from models.entropy_coding import *
from GPCC.gpcc_wrapper import *
from PCGCv2.eval import test_one_frame
import pandas as pd
import collections
import tqdm
from pytorch3d.ops import knn_points
import math
from model_utils import *
import torch
import psnr


# def log_string(string):
#     logger.info(string)
#     print(string)

def old_PSNR(pc1, pc2, n1):
    pc1, pc2 = pc1.to(torch.float32), pc2.to(torch.float32)
    dist1, knn1, _ = knn_points(pc1, pc2, K=4)
    dist2, knn2, _ = knn_points(pc2, pc1, K=4)
    mask1 = (dist1 == dist1[:, :, :1])
    mask2 = (dist2 == dist2[:, :, :1])
    dist = max(dist1[:, :, 0].mean(), dist2[:, :, 0].mean())
    cd = max(dist1[:, :, 0].sqrt().mean(), dist2[:, :, 0].sqrt().mean())
    d1_psnr = 10*math.log(3*1023*1023/dist)/math.log(10)
    knn1_ = knn1.reshape(-1)
    n1_src = (n1.unsqueeze(2).repeat(1, 1, 4, 1)*(mask1.unsqueeze(-1))).reshape(-1, 3)
    n2 = torch.zeros_like(pc2.squeeze(0), dtype=torch.float64)
    n2.index_add_(0, knn1_, n1_src)
    n2 = n2.reshape(1, -1, 3)

    n2_counter = torch.zeros(pc2.size()[1], dtype=torch.float32, device=pc2.device)
    counter_knn1 = knn1.reshape(-1)
    n1_counter_src = mask1.reshape(-1).to(torch.float32)
    n2_counter.index_add_(0, counter_knn1, n1_counter_src)
    n2_counter = n2_counter.reshape(1, -1, 1)
    n2_counter += 0.00000001

    n2 /= n2_counter

    v2 = index_points(pc1, knn2) - pc2.unsqueeze(2)
    n2_ = index_points(n1, knn2)
    n21 = (n2_*(mask2.unsqueeze(-1))).sum(dim=2) / (mask2.sum(dim=-1, keepdim=True))
    n2 += (n2_counter < 0.0001) * n21

    d2_ = (((v2*n2_).sum(dim=-1).square()*mask2).sum(dim=-1)/mask2.sum(dim=-1)).mean()
    v1 = index_points(pc2, knn1) - pc1.unsqueeze(2)
    n1_ = index_points(n2, knn1)
    d1_ = (((v1 * n1_).sum(dim=-1).square() * mask1).sum(dim=-1) / mask1.sum(dim=-1)).mean()
    dist_ = max(d1_, d2_)
    MAX_ENERGY = 59.7
    d2_psnr = 10*math.log(3*MAX_ENERGY*MAX_ENERGY/dist_)/math.log(10)
    return d1_psnr, d2_psnr, cd.item()

if __name__ == '__main__':
    device = torch.device('cuda')
    # logger = logging.getLogger("Model")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('./%s.txt' % args.log_name)
    # file_handler.se0Level(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    tmp_dir = args.tmp_dir
    # tmp_dir = './tmp_'+''.join(random.sample('0123456789', 10))
    tmp_dir_ = Path(tmp_dir)
    tmp_dir_.mkdir(exist_ok=True)
    results_dir = args.results_dir
    results_dir_ = Path(results_dir)
    results_dir_.mkdir(exist_ok=True)
    gpcc_bitstream_filename = os.path.join(tmp_dir, 'gpcc.bin')

    # load model
    # log_string('PARAMETER ...')
    # log_string(args)
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(channels=args.channels)
    model.eval()

    LOSSLESS_MODEL = importlib.import_module(args.lossless_model)
    lossless_model = LOSSLESS_MODEL.get_model()
    lossless_checkpoint = torch.load('./ddpcc_ckpts/lossless_coder.pth')
    old_paras = lossless_model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in lossless_checkpoint['model_state_dict'].items():
        k1 = k.replace('module.', '')
        if k1 in old_paras:
            new_state_dict[k1] = v
    old_paras.update(new_state_dict)
    lossless_model.load_state_dict(old_paras)
    lossless_model = lossless_model.to(device).eval()

    CKPTS = {
        'r3_0.10bpp.pth': 'r1.pth',
        'r4_0.15bpp.pth': 'r2.pth',
        'r5_0.25bpp.pth': 'r3.pth',
        'r6_0.3bpp.pth': 'r4.pth',
        'r7_0.4bpp.pth': 'r5.pth'
    }
    results = {
        "ckpt": [],
        "sequence": [],
        "frame_index": [],
        "points_number": [],
        "bits": [],
        "bpp": [],
        "d1_psnr": [],
        "d2_psnr": [],
        "q_level": []

    }
    Q_LEVELS = [7, 8, 9, 10, 11]
    SEQUENCES = [17]
    progress_bar = tqdm.tqdm(total=(args.frame_count - 1) * len(Q_LEVELS) * len(CKPTS))
    with torch.no_grad():
        for pcgcv2_ckpt_filename in CKPTS:
            # 读取模型
            ddpcc_ckpt = os.path.join(args.ckpt_dir, CKPTS[pcgcv2_ckpt_filename])
            pcgcv2_ckpt = os.path.join(args.pcgcv2_ckpt_dir, pcgcv2_ckpt_filename)
            checkpoint = torch.load(ddpcc_ckpt, map_location='cuda:0')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            # 这里只测第 17 个序列
            for sequence, q_level in product(SEQUENCES, Q_LEVELS):
                # 读取数据集
                dataset = Dataset(root_dir=args.dataset_dir, sequences=sequence)
                # d1_psnr_sum = 0
                # d2_psnr_sum = 0
                # bpp_sum = 0
                # bits_sum = 0
                # num_points_sum = 0
                # cd_sum = 0

                # encode the first frame
                points, feats, points_1, feats_1 = dataset[0]
                # 量化
                voxels, bbox_min, voxel_size = Dataset.quantify(points, q_level)
                voxels_1, bbox_min_1, voxel_size_1 = Dataset.quantify(points_1, q_level)
                # points, feats, points_1, feats_1 = collate_pointcloud_fn([])
                voxels, feats, voxels_1, feats_1 = collate_pointcloud_fn([(voxels, feats, voxels_1, feats_1)])

                f1 = ME.SparseTensor(features=feats, coordinates=voxels, device=device)
                bpp, d1_psnr, d2_psnr, f1 = test_one_frame(f1, pcgcv2_ckpt, os.path.join(tmp_dir,
                                                                                       'PCGCv2'))
                f1 = ME.SparseTensor(torch.ones_like(f1.F[:, :1]), coordinates=f1.C)
                # 收集结果
                # results['ckpt'].append(pcgcv2_ckpt)
                # results['sequence'].append(sequence)
                # results["bpp"].append(bpp)
                # results["d1_psnr"].append(d1psnr)
                # results["d2_psnr"].append(d2psnr)
                # results["points_number"].append(f1.size()[0] * 1.0)
                # results["bits"].append(f1.size()[0] * bpp)

                # log_string(str(0) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                # bpp_sum += bpp
                # d1_psnr_sum += d1psnr
                # d2_psnr_sum += d2psnr
                # num_points_sum += (f1.size()[0] * 1.0)
                # bits_sum += (f1.size()[0] * bpp)

                # 压缩第 1 到 args.frame_count 帧
                # 请注意变量 f1, f2 有如下的变化
                #     for i in ... do; f2 = dataset[i-1]; f1 = recon_f2; done
                for frame_index in range(1, args.frame_count):
                    out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                    # 从 dataset 中读取两帧点云
                    points, feats, points_1, feats_1 = dataset[frame_index-1]
                    # !!! 量化
                    voxels, bbox_min, voxel_size = Dataset.quantify(points, q_level)
                    voxels_1, bbox_min_1, voxel_size_1 = Dataset.quantify(points_1, q_level)
                    # 预处理
                    voxels, feats, voxels_1, feats_1 = collate_pointcloud_fn([(voxels, feats, voxels_1, feats_1)])

                    # 第 i - 1帧的稀疏张量
                    f2 = ME.SparseTensor(features=feats_1, coordinates=voxels_1, device=device)
                    num_points = f2.size()[0]

                    # 压缩第 i - 1, i 帧
                    # 这里用 model
                    ys2, out2, out_cls2, target2, keep2, ddpcc_bpp = model(f1, f2, f2.device)
                    ddpcc_bpp = ddpcc_bpp.item()
                    ys2_4_C = (ys2[4].C[:, 1:]//8).detach().cpu().numpy()
                    write_ply_data(os.path.join(tmp_dir, 'ys2_4.ply'), ys2_4_C)
                    gpcc_encode(os.path.join(tmp_dir, 'ys2_4.ply'), gpcc_bitstream_filename)

                    # encode ys2
                    # 这里用 lossless model
                    ys2_2 = ME.SparseTensor(torch.ones_like(ys2[2].F[:, :1]), coordinate_manager=ys2[2].coordinate_manager, coordinate_map_key=ys2[2].coordinate_map_key)
                    bits_ys2_2, quant_out2, cls, target = lossless_model.compressor(ys2_2, -1)
                    ys2_2_bpp = bits_ys2_2 / num_points
                    ys2_2_bpp = ys2_2_bpp.item()

                    gpcc_bpp = os.path.getsize(gpcc_bitstream_filename) * 8 / num_points
                    bpp = ddpcc_bpp + gpcc_bpp + ys2_2_bpp

                    # D1 D2
                    # write_ply_data(os.path.join(tmp_dir, 'f2.ply'), f2_C)

                    # 计算 f2 和 recon_f2 之间的 PSNR {
                    pc_ori = f2.C[:, 1:]
                    recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C)
                    pc_recon = recon_f2.C[:, 1:]
                    # { 用 open3d 来计算 pc_ori 的法向量，记作 n1
                    # pcd = open3d.geometry.PointCloud()
                    # pcd.points = open3d.utility.Vector3dVector(pc_ori.detach().cpu().numpy())
                    # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=5))
                    # n1 = torch.tensor(np.asarray(pcd.normals)).cuda()
                    # }
                    # pc_ori, pc_recon, n1 = pc_ori.unsqueeze(0), pc_recon.unsqueeze(0), n1.unsqueeze(0)
                    # pc_ori, pc_recon= pc_ori.unsqueeze(0), pc_recon.unsqueeze(0)
                    # !!! 反量化
                    # 比较 points_1 和 pc_recon 来计算 PSNR
                    # pc_recon 需要反量化
                    pc_recon = Dataset.unquantify(pc_recon, bbox_min_1, voxel_size_1)
                    d1_psnr = psnr.psnr_d1(points_1.cpu().numpy(), pc_recon.cpu().numpy(), max_energy=59.7)
                    d2_psnr = psnr.psnr_d2(points_1.cpu().numpy(), pc_recon.cpu().numpy(), max_energy=59.7)
                    # d1psnr, d2psnr, cd = old_PSNR(pc_ori, pc_recon, n1)
                    # log_string(str(i) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                    # }

                    # { 在这里把复原出来的 recon_f2 -> f1
                    # recon_f2 是未反量化的、解码的稀疏张量
                    f1 = recon_f2
                    # }

                    # 收集结果
                    results['ckpt'].append(CKPTS[pcgcv2_ckpt_filename])
                    results["frame_index"].append(frame_index)
                    results['sequence'].append(sequence)
                    results["bpp"].append(bpp)
                    results["d1_psnr"].append(d1_psnr)
                    results["d2_psnr"].append(d2_psnr)
                    results["points_number"].append(num_points * 1.0)
                    results["bits"].append(num_points * bpp)
                    # results["cd"].append(cd)
                    results["q_level"].append(q_level)

                    progress_bar.update(1)

                    # bpp_sum += bpp
                    # d1_psnr_sum += d1psnr
                    # d2_psnr_sum += d2psnr
                    # num_points_sum += (num_points * 1.0)
                    # cd_sum += cd

                # bpp_avg = bpp_sum / args.frame_count
                # d1_psnr_avg = d1_psnr_sum / args.frame_count
                # d2_psnr_avg = d2_psnr_sum / args.frame_count
                # cd_avg = cd_sum / args.frame_count
                # results[sequence_name]['bpp'].append(bpp_avg)
                # results[sequence_name]['d1-psnr'].append(d1_psnr_avg)
                # results[sequence_name]['d2-psnr'].append(d2_psnr_avg)
                # results[sequence_name]['cd'].append(cd_avg)
                # log_string(dataset.sequence_list[sequence] + ' average bpp: ' + str(bpp_avg))
                # log_string(dataset.sequence_list[sequence] + ' average d1-psnr: ' + str(d1_psnr_avg))
                # log_string(dataset.sequence_list[sequence] + ' average d2-psnr: ' + str(d2_psnr_avg))
                # log_string(dataset.sequence_list[sequence] + ' average cd: ' + str(cd_avg))

    progress_bar.close()
    pd.DataFrame(results).to_csv(os.path.join(results_dir, 'results.csv'), index=False)

    # for sequence_name in results:
    #     df = pd.DataFrame(results[sequence_name])
    #     df.to_csv(os.path.join(results_dir, sequence_name + '.csv'), index=False)
