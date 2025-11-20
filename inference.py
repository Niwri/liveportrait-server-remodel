# coding: utf-8

"""
The entrance of humans
"""

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.config.base_config import PrintableConfig, make_abs_path

class InferencePipeline():

    def __init__(self):
        tyro.extras.set_accent_color("bright_cyan")
        self.args = tyro.cli(ArgumentConfig)

        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
            )

        fast_check_args(self.args)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, self.args.__dict__)
        crop_cfg = partial_fields(CropConfig, self.args.__dict__)

        self.live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )

    def inference(self):
        if not self.live_portrait_pipeline:
            raise RuntimeError(
                "The inference pipeline is not yet set up"
            )

        path, _ = self.live_portrait_pipeline.execute(self.args)
        return path

    def specify_source_path(self, source_path):
        self.args.source = source_path

    def specify_output_path(self, output_path):
        self.args.output_dir = output_path

    def specify_driving_path(self, driving_path):
        self.args.driving = driving_path

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    pipeline = InferencePipeline()
    pipeline.live_portrait_pipeline.execute(args)

if __name__ == "__main__":
    main()
