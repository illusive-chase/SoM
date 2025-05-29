# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from rfstudio.engine.task import Task
from rfstudio.graphics import RGBImages
from rfstudio.io import dump_float32_image, load_float32_image
from rfstudio.ui import console
from seem.modeling import build_model as build_model_seem

# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem

# sam
from segment_anything import sam_model_registry
from semantic_sam import build_model

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES

from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.seem.tasks import inference_seem_pano
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto


@dataclass
class Script(Task):

    image_list: Path = Path('/root/autodl-tmp/data/yekai/dev/ArtGS/selected_data.txt')
    slider: float = 2.
    alpha: float = 0.7
    label_mode: Literal['Number', 'Alphabet'] = 'Number'
    output: Path = Path('data') / 'output'

    def inference(
        self,
        model_seem,
        model_sam,
        model_semsam,
        image: Path,
        output_path: Path,
        *,
        anno_mode,
    ) -> None:
        input_image = load_float32_image(image, alpha_color=(1, 1, 1))
        _image = Image.fromarray((input_image * 255).byte().numpy())

        slider = self.slider
        label_mode = self.label_mode
        alpha = self.alpha

        if slider < 1.5:
            model_name = 'seem'
        elif slider > 2.5:
            model_name = 'sam'
        else:
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]


        if label_mode == 'Alphabet':
            label_mode = 'a'
        else:
            label_mode = '1'

        text_size, hole_scale, island_scale=640,100,100
        text, text_part, text_thresh = '','','0.0'
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            semantic=False
            if model_name == 'semantic-sam':
                output, mask = inference_semsam_m2m_auto(model_semsam, _image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
            elif model_name == 'sam':
                output, mask = inference_sam_m2m_auto(model_sam, _image, text_size, label_mode, alpha, anno_mode)
            elif model_name == 'seem':
                output, mask = inference_seem_pano(model_seem, _image, text_size, label_mode, alpha, anno_mode)

        output = torch.from_numpy(output).float() / 255
        dump_float32_image(output_path, torch.cat((RGBImages(input_image).resize_to(output.shape[1], output.shape[0]).item(), output), dim=1))

    @torch.no_grad()
    def run(self) -> None:

        '''
        build args
        '''
        semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
        seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

        semsam_ckpt = "./swinl_only_sam_many2many.pth"
        sam_ckpt = "./sam_vit_h_4b8939.pth"
        seem_ckpt = "./seem_focall_v1.pt"

        opt_semsam = load_opt_from_config_file(semsam_cfg)
        opt_seem = load_opt_from_config_file(seem_cfg)
        opt_seem = init_distributed_seem(opt_seem)


        '''
        build model
        '''
        model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
        model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
        model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

        with self.image_list.open() as f:
            paths = [Path(x.rstrip()) for x in f.readlines() if x.rstrip()]

        self.output.mkdir(exist_ok=True, parents=True)
        with console.progress('Processing') as ptrack:
            for path in ptrack(paths):
                self.inference(model_seem, model_sam, model_semsam, path, self.output / (path.parent.parent.parent.parent.name + '.png'), anno_mode=['Mask', 'Mark'])

if __name__ == '__main__':
    Script(cuda=0).run()