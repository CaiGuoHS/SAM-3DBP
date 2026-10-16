import shutil
import logging
import os
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%y%m%d")
    return timestampDate + "-" + timestampTime


def compute_fp_fn_metrics(gt: np.ndarray, pred: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    gt = (gt > 0).astype(bool)
    pred = (pred > 0).astype(bool)

    # TP = np.sum(gt & pred)
    FP_voxel = np.sum((~gt) & pred)
    FN_voxel = np.sum(gt & (~pred))
    GT_voxel = np.sum(gt)

    single_voxel_vol = spacing[0] * spacing[1] * spacing[2]
    V_FP = FP_voxel * single_voxel_vol
    V_FN = FN_voxel * single_voxel_vol

    eps = 1e-6
    FP_ratio = FP_voxel / (GT_voxel + eps)
    FN_ratio = FN_voxel / (GT_voxel + eps)

    return V_FP, V_FN, FP_ratio, FN_ratio


def get_bp_region(mask_np, min_dist=5, max_dist=10):

    mask = np.squeeze(mask_np).astype(bool)

    if min_dist < 1:
        raise ValueError("min_dist must be greater than or equal to 1")

    if max_dist < min_dist:
        raise ValueError("max_dist must be greater than or equal to min_dist")

    dist = distance_transform_edt(~mask)
    outer_ring = (dist >= min_dist) & (dist <= max_dist)

    return outer_ring.astype(np.uint8)


def model_predict(img, coord, fpoints_coord, bpoints_coord, img_encoder, prompt_encoder, mask_decoder, input_size=128, embedding_size=32, handle_type=0, empty_point=True, device="cuda:0"):
    img = img.to(device)
    coord_x_min = coord[0][-1].start
    coord_y_min = coord[0][-2].start
    coord_z_min = coord[0][-3].start

    fpoints_coord_x = fpoints_coord[0]
    fpoints_coord_y = fpoints_coord[1]
    fpoints_coord_z = fpoints_coord[2]
    coord_fx = fpoints_coord_x - coord_x_min
    coord_fy = fpoints_coord_y - coord_y_min
    coord_fz = fpoints_coord_z - coord_z_min

    bpoints_coord_x = bpoints_coord[0]
    bpoints_coord_y = bpoints_coord[1]
    bpoints_coord_z = bpoints_coord[2]
    coord_bx = bpoints_coord_x - coord_x_min
    coord_by = bpoints_coord_y - coord_y_min
    coord_bz = bpoints_coord_z - coord_z_min

    is_empty = True
    fpoints_x = []
    fpoints_y = []
    fpoints_z = []
    for i in range(len(coord_fx)):
        if (0 <= coord_fx[i] < input_size) and (0 <= coord_fy[i] < input_size) and (0 <= coord_fz[i] < input_size):
            is_empty = False
            fpoints_x.append(coord_fx[i])
            fpoints_y.append(coord_fy[i])
            fpoints_z.append(coord_fz[i])

    bpoints_x = []
    bpoints_y = []
    bpoints_z = []
    if is_empty == False:
        for i in range(len(coord_bx)):
            if (0 <= coord_bx[i] < input_size) and (0 <= coord_by[i] < input_size) and (0 <= coord_bz[i] < input_size):
                bpoints_x.append(coord_bx[i])
                bpoints_y.append(coord_by[i])
                bpoints_z.append(coord_bz[i])

    if is_empty and empty_point:
        pred = torch.zeros_like(img[:, :1]).to(img. device)
    else:
        with torch.no_grad():

            image_embeddings_d, image_embeddings_h, image_embeddings_w = img_encoder(img)

            if is_empty:
                sparse_embeddings, dense_embeddings = prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                sparse_d_embeddings = sparse_embeddings.repeat(embedding_size, 1, 1)
                sparse_h_embeddings = sparse_embeddings.repeat(embedding_size, 1, 1)
                sparse_w_embeddings = sparse_embeddings.repeat(embedding_size, 1, 1)
                dense_d_embeddings = dense_embeddings.repeat(embedding_size, 1, 1, 1)
                dense_h_embeddings = dense_embeddings.repeat(embedding_size, 1, 1, 1)
                dense_w_embeddings = dense_embeddings.repeat(embedding_size, 1, 1, 1)
            else:
                num_fsamples = len(fpoints_x)
                num_bsamples = len(bpoints_x)

                fx = np.array(fpoints_x)
                fy = np.array(fpoints_y)
                fz = np.array(fpoints_z)
                bx = np.array(bpoints_x)
                by = np.array(bpoints_y)
                bz = np.array(bpoints_z)

                points_d = [[fx[i], fy[i]] for i in range(num_fsamples)]
                points_h = [[fz[i], fx[i]] for i in range(num_fsamples)]
                points_w = [[fy[i], fz[i]] for i in range(num_fsamples)]
                points_d.extend([[bx[i], by[i]] for i in range(num_bsamples)])
                points_h.extend([[bz[i], bx[i]] for i in range(num_bsamples)])
                points_w.extend([[by[i], bz[i]] for i in range(num_bsamples)])
                points_d_torch = torch.tensor(points_d, dtype=torch.float32, device=device).unsqueeze(0).repeat(embedding_size, 1, 1)
                points_h_torch = torch.tensor(points_h, dtype=torch.float32, device=device).unsqueeze(0).repeat(embedding_size, 1, 1)
                points_w_torch = torch.tensor(points_w, dtype=torch.float32, device=device).unsqueeze(0).repeat(embedding_size, 1, 1)

                labels = [1 for _ in range(num_fsamples)]
                labels.extend([-1 for _ in range(num_bsamples)])
                labels_d_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0).repeat(embedding_size, 1)
                labels_h_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0).repeat(embedding_size, 1)
                labels_w_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0).repeat(embedding_size, 1)

                if handle_type == 0:
                    for i in range(num_bsamples):
                        b_idx = num_fsamples + i
                        labels_d_torch[int(bz[i] // 4), b_idx] = 0
                        labels_h_torch[int(by[i] // 4), b_idx] = 0
                        labels_w_torch[int(bx[i] // 4), b_idx] = 0
                elif handle_type == 1:
                    for i in range(num_bsamples):
                        b_idx = num_fsamples + i
                        labels_d_torch[:, b_idx] = 0
                        labels_h_torch[:, b_idx] = 0
                        labels_w_torch[:, b_idx] = 0
                elif handle_type == 2:
                    for i in range(num_bsamples):
                        b_idx = num_fsamples + i
                        bz_center = int(bz[i] // 4)
                        bz_neighbor = np.arange(bz_center - 1, bz_center + 2)
                        bz_neighbor = np.clip(bz_neighbor, 0, embedding_size - 1)
                        by_center = int(by[i] // 4)
                        by_neighbor = np.arange(by_center - 1, by_center + 2)
                        by_neighbor = np.clip(by_neighbor, 0, embedding_size - 1)
                        bx_center = int(bx[i] // 4)
                        bx_neighbor = np.arange(bx_center - 1, bx_center + 2)
                        bx_neighbor = np.clip(bx_neighbor, 0, embedding_size - 1)
                        labels_d_torch[bz_neighbor, b_idx] = 0
                        labels_h_torch[by_neighbor, b_idx] = 0
                        labels_w_torch[bx_neighbor, b_idx] = 0
                else:
                    for i in range(num_bsamples):
                        b_idx = num_fsamples + i
                        labels_d_torch[int(bz[i] // 4), b_idx] = 0
                        labels_h_torch[int(by[i] // 4), b_idx] = 0
                        labels_w_torch[int(bx[i] // 4), b_idx] = 0

                sparse_d_embeddings, dense_d_embeddings = prompt_encoder(
                    points=(points_d_torch, labels_d_torch),
                    boxes=None,
                    masks=None,
                )
                sparse_h_embeddings, dense_h_embeddings = prompt_encoder(
                    points=(points_h_torch, labels_h_torch),
                    boxes=None,
                    masks=None,
                )
                sparse_w_embeddings, dense_w_embeddings = prompt_encoder(
                    points=(points_w_torch, labels_w_torch),
                    boxes=None,
                    masks=None,
                )

            pred = mask_decoder(
                image_embeddings=(image_embeddings_d, image_embeddings_h, image_embeddings_w),
                image_pe=prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=(sparse_d_embeddings, sparse_h_embeddings, sparse_w_embeddings),
                dense_prompt_embeddings=(dense_d_embeddings, dense_h_embeddings, dense_w_embeddings),
            )

    return pred.cpu()