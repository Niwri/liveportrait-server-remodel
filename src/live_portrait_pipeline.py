# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
import time
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load, load_image_rgb_bytes
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth
from .utils.rprint import rlog as log
# from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class LivePortraitPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

        self.processed_source = None

    def process_source(self, img_bytes):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        flag_normalize_lip = inf_cfg.flag_normalize_lip

        img_rgb = load_image_rgb_bytes(img_bytes)
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
        source_rgb_lst = [img_rgb]

        if inf_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
            if crop_info is None:
                raise Exception("No face detected in the source image!")
            source_lmk = crop_info['lmk_crop']
            img_crop_256x256 = crop_info['img_crop_256x256']
        else:
            source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
            img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
            if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
                lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        mask_ori_float = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

        self.processed_source = {
            'source_rgb_lst': source_rgb_lst,
            'img_crop_256x256': img_crop_256x256,
            'crop_info': crop_info,
            'source_lmk': source_lmk,
            'x_s_info': x_s_info,
            'x_c_s': x_c_s,
            'R_s': R_s,
            'f_s': f_s,
            'x_s': x_s,
            'lip_delta_before_animation': lip_delta_before_animation,
            'mask_ori_float': mask_ori_float
        }

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device = self.live_portrait_wrapper.device

        ######## load source input ########
        if self.processed_source is None:
            raise Exception(f"Source has not yet been processed")

        source_rgb_lst = self.processed_source['source_rgb_lst']
        img_crop_256x256 = self.processed_source['img_crop_256x256']
        crop_info = self.processed_source['crop_info']
        source_lmk = self.processed_source['source_lmk']
        x_s_info = self.processed_source['x_s_info']
        x_c_s = self.processed_source['x_c_s']
        R_s = self.processed_source['R_s']
        f_s = self.processed_source['f_s']
        x_s = self.processed_source['x_s']
        lip_delta_before_animation = self.processed_source['lip_delta_before_animation']
        mask_ori_float = self.processed_source['mask_ori_float']

        flag_load_from_template = is_template(args.driving)
        if not flag_load_from_template:
            raise Exception(f"Not a pkl file!")

        driving_rgb_crop_256x256_lst = None

        # NOTE: load from template, it is fast, but the cropping video is None
        log(f"Load from template: {args.driving}, NOT the video, so the cropping video and audio are both NULL.", style='bold green')
        driving_template_dct = load(args.driving)
        c_d_eyes_lst = driving_template_dct['c_eyes_lst'] if 'c_eyes_lst' in driving_template_dct.keys() else driving_template_dct['c_d_eyes_lst'] # compatible with previous keys
        c_d_lip_lst = driving_template_dct['c_lip_lst'] if 'c_lip_lst' in driving_template_dct.keys() else driving_template_dct['c_d_lip_lst']

        ######## prepare for pasteback ########
        startTime = time.time()
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        I_p_lst = []
        R_d_0, x_d_0_info = None, None
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        lip_delta_before_animation, eye_delta_before_animation = None, None


        log(f"The output of image-driven portrait animation is an image.")
        x_d_i_info = dct2device(driving_template_dct['motion'], device)
        R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

        R_d_0 = R_d_i
        x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()
        if inf_cfg.flag_relative_motion:
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            else:
                R_new = R_s
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            if inf_cfg.animation_region == "all":
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
            else:
                scale_new = x_s_info['scale']
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                t_new = x_s_info['t']
        else:
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                R_new = R_d_i
            else:
                R_new = R_s
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            scale_new = x_s_info['scale']
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                t_new = x_d_i_info['t']
            else:
                t_new = x_s_info['t']

        # print("A", time.time()-startTime)
        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        startTime = time.time()
        # Algorithm 1:
        if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            # print("1")
            # without stitching or retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new += lip_delta_before_animation
            if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                x_d_i_new += eye_delta_before_animation
            else:
                pass
        elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            # print("2")
            # with stitching and without retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
            else:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                x_d_i_new += eye_delta_before_animation
        else:
            # print("3")
            eyes_delta, lip_delta = None, None
            if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst
                combined_eye_ratio_tensor = self.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                eyes_delta = self.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
            if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                c_d_lip_i = c_d_lip_lst
                combined_lip_ratio_tensor = self.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

            if inf_cfg.flag_relative_motion:  # use x_s
                x_d_i_new = x_s + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)
            else:  # use x_d,i
                x_d_i_new = x_d_i_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if inf_cfg.flag_stitching:
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)

        # print("B", time.time()-startTime)
        startTime = time.time()
        x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
        I_p_lst.append(I_p_i)

        # print("C", time.time()-startTime) # Taking the longest by far
        startTime = time.time()
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
            # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
            I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
            I_p_pstbk_lst.append(I_p_pstbk)
        # print("C", time.time()-startTime)
        mkdir(args.output_dir)
        wfp_concat = None
        ######### build the final concatenation result #########
        # driving frame | source frame | generation
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)


        wfp_concat = osp.join(args.output_dir, f'{basename(args.driving)}_concat.jpg')
        cv2.imwrite(wfp_concat, frames_concatenated[0][..., ::-1])
        wfp = osp.join(args.output_dir, f'{basename(args.driving)}.jpg')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            cv2.imwrite(wfp, I_p_pstbk_lst[0][..., ::-1])
        else:
            cv2.imwrite(wfp, frames_concatenated[0][..., ::-1])
        # final log
        log(f'Animated image: {wfp}')
        log(f'Animated image with concat: {wfp_concat}')

        return wfp, wfp_concat
