import os

from monai.data import NibabelWriter

from segment_anything import sam_model_registry

import argparse
import logging
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference

from modeling.image_encoder_sam3dbp import ImageEncoderViT as ImageEncoderViT
from modeling.mask_decoder_sam3dbp import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PromptEncoder

from utils.datasets import load_data_volume
from utils.util import model_predict, get_bp_region, compute_fp_fn_metrics
from utils.util import setup_logger
import utils.SurfaceDistance as SD


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

    parser.add_argument("--input_num_points", default=(5, 5), type=tuple)
    parser.add_argument("--input_num_fpoints", default=3, type=int)
    parser.add_argument("--input_num_bpoints", default=1, type=int)
    parser.add_argument("--input_gamma", default=(1, 10), type=int)
    parser.add_argument("--handle_type", default=0, type=int)
    parser.add_argument("--num_times", default=5, type=int)
    parser.add_argument("--save_masks", default=True, type=bool)

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
    parser.add_argument("--num_worker", default=2, type=int)
    parser.add_argument("--checkpoint", default="best", type=str)
    parser.add_argument("--tolerance", default=5, type=int)
    args = parser.parse_args()
    rngf = np.random.default_rng()
    rngb = np.random.default_rng()
    if args.seed >= 0:
        # np.random.seed(args.seed)
        rngf = np.random.default_rng(args.seed)
        rngb = np.random.default_rng(args.seed)
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device

    save_path = os.path.join(args.save_dir, args.data, f"sam3dbp_{args.num_fpoints}_{args.num_bpoints}_{args.gamma}")
    setup_logger(logger_name=f"{args.data}_{args.num_fpoints}_{args.num_bpoints}_{args.gamma}_{args.checkpoint}_{args.input_num_points}_{args.input_num_fpoints}_{args.input_num_bpoints}_{args.input_gamma}_{args.num_times}_{args.handle_type}", root=save_path, screen=True, tofile=True)
    logger = logging.getLogger(f"{args.data}_{args.num_fpoints}_{args.num_bpoints}_{args.gamma}_{args.checkpoint}_{args.input_num_points}_{args.input_num_fpoints}_{args.input_num_bpoints}_{args.input_gamma}_{args.num_times}_{args.handle_type}")
    logger.info(str(args))
    input_size = args.rand_crop_size[0]

    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_dir+args.data,
        augmentation=False,
        split="test",
        convert_to_sam=False,
        deterministic=True,
        num_worker=args.num_worker
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
    img_encoder.load_state_dict(sam.image_encoder.state_dict(), strict=False)
    img_encoder.to(device)
    img_encoder.update_embed()

    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(args.embedding_size, args.embedding_size),
        input_image_size=(input_size, input_size),
        mask_in_chans=16,
    )
    prompt_encoder.load_state_dict(sam.prompt_encoder.state_dict())
    prompt_encoder.to(device)

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
    mask_decoder.load_state_dict(torch.load(os.path.join(save_path, file), map_location='cpu')["decoder_dict"], strict=True)
    mask_decoder.to(device)

    del sam
    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    dice_loss = DiceLoss(squared_pred=True, reduction="mean")

    dice_total = []
    nsd_total = []
    fp_voxel_total = []
    fn_voxel_total = []
    fp_radio_total = []
    fn_radio_total = []
    for n in range(args.num_times):
        if args.save_masks:
            output_path = os.path.join(save_path, "_".join([str.split(str.split(logger.handlers[0].baseFilename, "/")[-1], ".")[0], str(n)]))
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        with torch.no_grad():
            loss_summary = []
            loss_nsd = []
            
            fp_voxel_summary = []
            fn_voxel_summary = []
            fp_radio_summary = []
            fn_radio_summary = []
            
            for idx, (img, seg, spacing, path, affine) in enumerate(test_data):
                logger.info(f"seg: {seg.sum()}, {path}, {img.shape}, {seg.shape}, {spacing}")
                seg = seg.unsqueeze(0)
                seg_img = F.interpolate(seg, size=img.shape[2:], mode='nearest').detach().cpu().numpy()
                # img = img.to(device)

                num_fsamples = args.input_num_fpoints
                num_bsamples = args.input_num_bpoints

                fp_region = np.squeeze(seg_img)
                fpoints_sub = np.argwhere(fp_region == 1)
                fpoints_num = len(fpoints_sub)
                need_fp_num = min(num_fsamples, fpoints_num)

                fsamples = rngf.choice(fpoints_num, min(args.input_num_points[0], fpoints_num), replace=False)
                fpoints_z = fpoints_sub[fsamples[:need_fp_num]][:, -3]
                fpoints_y = fpoints_sub[fsamples[:need_fp_num]][:, -2]
                fpoints_x = fpoints_sub[fsamples[:need_fp_num]][:, -1]

                bpoints_z = []
                bpoints_y = []
                bpoints_x = []
                if num_bsamples > 0:
                    bp_region = get_bp_region(seg_img, args.input_gamma[0], args.input_gamma[1])
                    bpoints_sub = np.argwhere(bp_region == 1)
                    bpoints_num = len(bpoints_sub)
                    need_bp_num = min(num_bsamples, bpoints_num)
                    bsamples = rngb.choice(bpoints_num, min(args.input_num_points[1], bpoints_num), replace=False)
                    bpoints_x = bpoints_sub[bsamples[:need_bp_num]][:, -1]
                    bpoints_y = bpoints_sub[bsamples[:need_bp_num]][:, -2]
                    bpoints_z = bpoints_sub[bsamples[:need_bp_num]][:, -3]

                logger.info((fpoints_x, fpoints_y, fpoints_z))
                logger.info((bpoints_x, bpoints_y, bpoints_z))

                fx = np.array(fpoints_x)
                fy = np.array(fpoints_y)
                fz = np.array(fpoints_z)
                bx = np.array(bpoints_x)
                by = np.array(bpoints_y)
                bz = np.array(bpoints_z)
                fpoints_coord = (fx, fy, fz)
                bpoints_coord = (bx, by, bz)

                logger.info((fx * seg.shape[-1] / img.shape[-1], fy * seg.shape[-2] / img.shape[-2], fz * seg.shape[-3] / img.shape[-3]))
                logger.info((bx * seg.shape[-1] / img.shape[-1], by * seg.shape[-2] / img.shape[-2], bz * seg.shape[-3] / img.shape[-3]))

                pred = sliding_window_inference(img, [input_size, input_size, input_size], overlap=0.5, sw_batch_size=1,
                                                mode="gaussian",
                                                sigma_scale=0.25,
                                                device=torch.device("cpu"),
                                                predictor=partial(model_predict,
                                                                  fpoints_coord=fpoints_coord,
                                                                  bpoints_coord=bpoints_coord,
                                                                  img_encoder=img_encoder,
                                                                  prompt_encoder=prompt_encoder,
                                                                  mask_decoder=mask_decoder,
                                                                  input_size=input_size,
                                                                  embedding_size=args.embedding_size,
                                                                  handle_type=args.handle_type,
                                                                  ),
                                                with_coord=True)

                final_pred = F.interpolate(pred, size=seg.shape[2:], mode="trilinear")
                final_pred = torch.sigmoid(final_pred)
                masks = (final_pred > 0.5).float()
                loss = 1 - dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                
                ssd = SD.compute_surface_distances((seg == 1)[0, 0].cpu().numpy(), (masks==1)[0, 0].cpu().numpy(), spacing_mm=spacing[0].numpy())
                nsd = SD.compute_surface_dice_at_tolerance(ssd, args.tolerance)
                loss_nsd.append(nsd)

                fp_voxel, fn_voxel, fp_radio, fn_radio = compute_fp_fn_metrics(seg.detach().cpu().numpy(), masks.detach().cpu().numpy(), spacing=spacing[0].numpy())
                fp_voxel_summary.append(fp_voxel)
                fn_voxel_summary.append(fn_voxel)
                fp_radio_summary.append(fp_radio)
                fn_radio_summary.append(fn_radio)
                
                logger.info(f" Case {test_data.dataset.img_dict[idx]} - Dice {loss.item():.6f} | NSD {nsd:.6f}")
                logger.info(f"fp_voxel: {fp_voxel:.2f}, fn_voxel: {fn_voxel:.2f}, fp_radio: {fp_radio:.6f}, fn_radio: {fn_radio:.6f}")

                if args.save_masks:
                    name = path[0].split("/")[-2] + path[0].split("/")[-1]
                    writer = NibabelWriter()
                    if args.data != "kits":
                        image_data_pre = np.uint8(masks.squeeze(0).squeeze(0).permute(2, 1, 0).detach().cpu())
                    else:
                        image_data_pre = np.uint8(masks.squeeze(0).squeeze(0).detach().cpu())
                    affine = affine.squeeze(0).detach().cpu().numpy()
                    writer.set_data_array(image_data_pre, channel_dim=None)
                    writer.set_metadata({"affine": affine})
                    writer.write(f"{output_path}/{args.data}_pred_{name}", verbose=True)

            dice_total.append(np.mean(loss_summary))
            nsd_total.append(np.mean(loss_nsd))
            fp_voxel_total.append(np.mean(fp_voxel_summary))
            fn_voxel_total.append(np.mean(fn_voxel_summary))
            fp_radio_total.append(np.mean(fp_radio_summary))
            fn_radio_total.append(np.mean(fn_radio_summary))
            logger.info(f"- Test metrics Dice: {np.mean(loss_summary):.6f}")
            logger.info(f"- Test metrics NSD: {np.mean(loss_nsd):.6f}")
            logger.info(f"- Test metrics fp_voxel: {np.mean(fp_voxel_summary):.6f}")
            logger.info(f"- Test metrics fn_voxel: {np.mean(fn_voxel_summary):.6f}")
            logger.info(f"- Test fp_radio: {np.mean(fp_radio_summary):.6f}")
            logger.info(f"- Test fn_radio: {np.mean(fn_radio_summary):.6f}")

    for n in range(args.num_times):
        logger.info(f"\t{dice_total[n]:.6f}\t{nsd_total[n]:.6f}")
    logger.info(f"\t{np.mean(dice_total):.6f}\t{np.mean(nsd_total):.6f}")
    
    for n in range(args.num_times):
        logger.info(f"\t{fp_voxel_total[n]:.6f}\t{fn_voxel_total[n]:.6f}")
    logger.info(f"\t{np.mean(fp_voxel_total):.6f}\t{np.mean(fn_voxel_total):.6f}")

    for n in range(args.num_times):
        logger.info(f"\t{fp_radio_total[n]:.6f}\t{fn_radio_total[n]:.6f}")
    logger.info(f"\t{np.mean(fp_radio_total):.6f}\t{np.mean(fn_radio_total):.6f}")


if __name__ == "__main__":
    main()
