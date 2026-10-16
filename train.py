import os

from monai.inferers import sliding_window_inference

import argparse
import logging
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss

from segment_anything import sam_model_registry
from modeling.image_encoder_sam3dbp import ImageEncoderViT
from modeling.mask_decoder_sam3dbp import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PromptEncoder

from utils.datasets import load_data_volume
from utils.util import save_checkpoint, model_predict, get_bp_region
from utils.util import setup_logger


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--num_fpoints", default=3, type=int)
    parser.add_argument("--num_bpoints", default=3, type=int)
    parser.add_argument("--embedding_size", default=32, type=int)
    parser.add_argument("--new_window_size", default=16, type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--beta", default=0.8, type=float)
    parser.add_argument("--gamma", default=(1, 10), type=tuple)
    parser.add_argument(
        "--data", default="colon", type=str, choices=["colon", "lits", "kits", "pancreas"]
    )
    parser.add_argument(
        "--data_dir",
        default="./datasets/",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="ckpts",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=(128, 128, 128),
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_epoch", default=200, type=int)
    parser.add_argument("--num_worker", default=4, type=int)

    args = parser.parse_args()
    device = args.device
    save_path = os.path.join(args.save_dir, args.data, f"sam3dbp_{args.num_fpoints}_{args.num_bpoints}_{args.gamma}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logger(logger_name="train", root=save_path, screen=True, tofile=True)
    logger = logging.getLogger("train")
    logger.info(str(args))
    setup_logger(logger_name="train_loss", root=save_path, screen=True, tofile=True)
    logger_loss = logging.getLogger("train_loss")
    input_size = args.rand_crop_size[0]

    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_dir+args.data,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker,
    )
    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_dir+args.data,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        do_val_crop=False,
        num_worker = args.num_worker
    )
    sam = sam_model_registry["vit_b"](checkpoint="ckpts/sam_vit_b_01ec64.pth")

    img_encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
        embedding_size=args.embedding_size,
        new_window_size=args.new_window_size,
    )

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(args.embedding_size, args.embedding_size),
        input_image_size=(input_size, input_size),
        mask_in_chans=16,
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
    )

    img_encoder.load_state_dict(sam.image_encoder.state_dict(), strict=False)
    prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict())
    mask_decoder.iou_token.weight = sam.mask_decoder.iou_token.weight
    mask_decoder.mask_tokens.weight = sam.mask_decoder.mask_tokens.weight
    mask_decoder.transformer.load_state_dict(sam.mask_decoder.transformer.state_dict())
    mask_decoder.output_upscaling.load_state_dict(sam.mask_decoder.output_upscaling.state_dict())
    mask_decoder.output_hypernetworks_mlp.load_state_dict(sam.mask_decoder.output_hypernetworks_mlps[0].state_dict())

    del sam

    for param in img_encoder.parameters():
        param.requires_grad = False

    for param in prompt_encoder.parameters():
        param.requires_grad = False

    img_encoder.to(device)
    prompt_encoder.to(device)
    mask_decoder.to(device)
    img_encoder.update_embed()

    decoder_opt = torch.optim.AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0.00001)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.1, total_iters=args.max_epoch)

    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    dice_loss = DiceLoss(squared_pred=True, reduction="mean")
    best_loss = np.inf

    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()

    for epoch_num in range(args.max_epoch):
        loss_summary = []
        mask_decoder.train()
        for idx, (img, seg, spacing, path, affine) in enumerate(train_data):
            logger.info(f"seg: {seg.sum()}, {path}, {img.shape}, {seg.shape}")

            with torch.no_grad():
                image_embeddings_d, image_embeddings_h, image_embeddings_w = img_encoder(img.to(device))

                num_fsamples = torch.randint(low=1, high=args.num_fpoints + 1, size=()).item()
                num_bsamples = torch.randint(high=args.num_bpoints + 1, size=()).item()

                fp_region = np.squeeze(seg)
                fpoints_sub = np.argwhere(fp_region == 1)
                fpoints_num = len(fpoints_sub)
                logger.info(f"fpoints: {fpoints_num}")
                if fpoints_num > 0:
                    fsamples = np.random.choice(fpoints_num, min(num_fsamples, fpoints_num), replace=False)
                    num_fsamples = len(fsamples)
                    fpoints_x = fpoints_sub[fsamples][:, -1]
                    fpoints_y = fpoints_sub[fsamples][:, -2]
                    fpoints_z = fpoints_sub[fsamples][:, -3]

                    bpoints_z = []
                    bpoints_y = []
                    bpoints_x = []
                    if num_bsamples > 0:
                        bp_region = get_bp_region(seg, args.gamma[0], args.gamma[1])
                        bpoints_sub = np.argwhere(bp_region == 1)
                        b_expand = bpoints_sub[:, np.newaxis, :]
                        f_expand = fpoints_sub[fsamples[:num_fsamples]][np.newaxis, :, :]
                        coord_eq = (b_expand == f_expand)
                        match_count = coord_eq.sum(axis=2)
                        is_overlap = (match_count >= 2).any(axis=1)
                        bpoints_sub = bpoints_sub[~is_overlap]
                        bpoints_num = len(bpoints_sub)
                        logger.info(f"bpoints: {bpoints_num}")
                        if bpoints_num > 0:
                            bsamples = np.random.choice(bpoints_num, min(num_bsamples, bpoints_num), replace=False)
                            num_bsamples = len(bsamples)
                            bpoints_x = bpoints_sub[bsamples][:, -1]
                            bpoints_y = bpoints_sub[bsamples][:, -2]
                            bpoints_z = bpoints_sub[bsamples][:, -3]

                    fx = np.array(fpoints_x)
                    fy = np.array(fpoints_y)
                    fz = np.array(fpoints_z)
                    bx = np.array(bpoints_x)
                    by = np.array(bpoints_y)
                    bz = np.array(bpoints_z)

                    logger.info((fx, fy, fz))
                    logger.info((bx, by, bz))

                    points_d = [[fx[i], fy[i]] for i in range(num_fsamples)]
                    points_h = [[fz[i], fx[i]] for i in range(num_fsamples)]
                    points_w = [[fy[i], fz[i]] for i in range(num_fsamples)]
                    points_d.extend([[bx[i], by[i]] for i in range(num_bsamples)])
                    points_h.extend([[bz[i], bx[i]] for i in range(num_bsamples)])
                    points_w.extend([[by[i], bz[i]] for i in range(num_bsamples)])
                    points_d_torch = torch.tensor(points_d, dtype=torch.float32, device=device).unsqueeze(0).repeat(args.embedding_size, 1, 1)
                    points_h_torch = torch.tensor(points_h, dtype=torch.float32, device=device).unsqueeze(0).repeat(args.embedding_size, 1, 1)
                    points_w_torch = torch.tensor(points_w, dtype=torch.float32, device=device).unsqueeze(0).repeat(args.embedding_size, 1, 1)
                    
                    labels = [1 for _ in range(num_fsamples)]
                    labels.extend([-1 for _ in range(num_bsamples)])
                    labels_d_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0).repeat(args.embedding_size, 1)
                    labels_h_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0).repeat(args.embedding_size, 1)
                    labels_w_torch = torch.tensor(labels, dtype=torch.int32, device=device).unsqueeze(0).repeat(args.embedding_size, 1)

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
                else:
                    sparse_embeddings_none, dense_embeddings_none = prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    sparse_d_embeddings = sparse_embeddings_none.repeat(args.embedding_size, 1, 1)
                    sparse_h_embeddings = sparse_embeddings_none.repeat(args.embedding_size, 1, 1)
                    sparse_w_embeddings = sparse_embeddings_none.repeat(args.embedding_size, 1, 1)
                    dense_d_embeddings = dense_embeddings_none.repeat(args.embedding_size, 1, 1, 1)
                    dense_h_embeddings = dense_embeddings_none.repeat(args.embedding_size, 1, 1, 1)
                    dense_w_embeddings = dense_embeddings_none.repeat(args.embedding_size, 1, 1, 1)

            masks = mask_decoder(
                image_embeddings=(image_embeddings_d, image_embeddings_h, image_embeddings_w),
                image_pe=prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=(sparse_d_embeddings, sparse_h_embeddings, sparse_w_embeddings),
                dense_prompt_embeddings=(dense_d_embeddings, dense_h_embeddings, dense_w_embeddings),
            )

            seg = seg.to(device)
            loss = args.alpha * bce_loss(masks, seg.float()) + args.beta * seg_loss(masks, seg)

            loss_summary.append(loss.detach().cpu().numpy())
            decoder_opt.zero_grad()
            loss.backward()
            logger.info('epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(loss_summary[-1].flatten()[0]))
            decoder_opt.step()
        decoder_scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))
        logger_loss.info("Train " + str(np.mean(loss_summary)))


        if epoch_num >= 180:
            mask_decoder.eval()
            with torch.no_grad():
                loss_summary = []
                rngf = np.random.default_rng()
                if args.seed >= 0:
                    rngf = np.random.default_rng(args.seed)
                for idx, (img, seg, spacing, path, affine) in enumerate(val_data):
                    logger.info(f"seg: {seg.sum()}, {path}, {img.shape}, {seg.shape}")
                    seg = seg.unsqueeze(0)
                    seg_img = F.interpolate(seg, size=img.shape[2:], mode='nearest').detach().cpu().numpy()

                    num_fsamples = args.num_fpoints

                    fp_region = np.squeeze(seg_img)
                    fpoints_sub = np.argwhere(fp_region == 1)
                    fpoints_num = len(fpoints_sub)
                    need_fp_num = min(num_fsamples, fpoints_num)

                    fsamples = rngf.choice(fpoints_num, need_fp_num, replace=False)
                    fpoints_z = fpoints_sub[fsamples][:, -3]
                    fpoints_y = fpoints_sub[fsamples][:, -2]
                    fpoints_x = fpoints_sub[fsamples][:, -1]

                    logger.info((fpoints_x, fpoints_y, fpoints_z))

                    fx = np.array(fpoints_x)
                    fy = np.array(fpoints_y)
                    fz = np.array(fpoints_z)
                    bx = np.array([])
                    by = np.array([])
                    bz = np.array([])
                    fpoints_coord = (fx, fy, fz)
                    bpoints_coord = (bx, by, bz)

                    logger.info((fx * seg.shape[-1] / img.shape[-1], fy * seg.shape[-2] / img.shape[-2], fz * seg.shape[-3] / img.shape[-3]))

                    pred = sliding_window_inference(img, [input_size, input_size, input_size], overlap=0.5, sw_batch_size=1,
                                                    mode="gaussian",
                                                    sigma_scale=0.25,
                                                    # device=device,
                                                    device=torch.device("cpu"),
                                                    predictor=partial(model_predict,
                                                                      fpoints_coord=fpoints_coord,
                                                                      bpoints_coord=bpoints_coord,
                                                                      img_encoder=img_encoder,
                                                                      prompt_encoder=prompt_encoder,
                                                                      mask_decoder=mask_decoder,
                                                                      input_size=input_size,
                                                                      embedding_size=args.embedding_size,
                                                                      ),
                                                    with_coord=True)

                    final_pred = F.interpolate(pred, size=seg.shape[2:], mode="trilinear")
                    final_pred = F.sigmoid(final_pred)
                    masks = (final_pred > 0.5).float()
                    loss = dice_loss(masks, seg)
                    loss_summary.append(loss.detach().cpu().numpy())
                    logger.info('epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(loss_summary[-1].flatten()[0]))

            logger.info("- Val metrics: " + str(np.mean(loss_summary)))
            logger_loss.info("Val " + str(np.mean(loss_summary)))

            is_best = False
            if np.mean(loss_summary) < best_loss:
                best_loss = np.mean(loss_summary)
                is_best = True
                logger.info(f"Evolution best: {epoch_num}")

            save_checkpoint({
                             "decoder_dict": mask_decoder.state_dict(),
                             },
                            is_best=is_best,
                            checkpoint=save_path)
            logger.info("- Val metrics best: " + str(best_loss))
        else:
            save_checkpoint({
                             "decoder_dict": mask_decoder.state_dict(),
                             },
                            is_best=False,
                            checkpoint=save_path)
            logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()

