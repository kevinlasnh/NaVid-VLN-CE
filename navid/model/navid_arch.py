#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2023 Yanwei Li
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


from navid.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    NAVIGATION_SPECIAL_TOKEN,
    NAVIGATION_IDENTIFIER,
    IAMGE_SEPARATOR,
)


class NaVidMetaModel:
    def __init__(self, config):
        # 虽然这里代码没有明确写要被调用的父类，但是由于其现在的对象实际是 LlavaAttLlamaModel ，所以其会从 NaVidMetaModel 后面继续沿着当前对象的继承顺序寻找下一个初始化函数
        super(NaVidMetaModel, self).__init__(config)

        # 初始化多模态组件
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, 'image_processor', None)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_attention_modules(self, model_args, for_eval=False):
        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        pretrain_qformer = getattr(model_args, "pretrain_qformer", None)
        self.config.compress_type = getattr(model_args, "compress_type", None)


class NaVidMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # 统一接收两种视觉输入
    def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
        # 判断 images 是不是已经是预计算特征
        if long_video:
            # use pre-computed features
            image_features = images
        else:
            # (n, 3, 224, 224)
            # 原始图片先通过视觉塔
            # 这一步将原始图片编码成为视觉特征，输出的维度是 (n, 257, 1408)，其中 257 是视觉 token 的数量，1408 是每个 token 的特征维度
            # 还没有将原始的视觉特征映射到 Llama 的隐藏维度
            # 这个函数调用完就把 RGB 图像帧转化成视觉 token 了
            image_features = self.get_model().get_vision_tower()(images)
            # (n, 257, 1408)

        # 虽然这里写的是 attention 但是实际上不是 attention 操作，这个函数
        # 其最主要的操作还是将图像特征映射到了 Llama 的隐藏维度，并且根据 compress_type 对视觉 token 进行了压缩
        image_features, video_or_not, nav_or_not = self.vlm_attention(
            image_features,
            prompts=prompts,
            image_counts=image_counts,
            long_video=long_video,
        )
        return image_features, video_or_not, nav_or_not

    # 把已经由视觉塔提取好的，混在一起的视觉特征，重新按样本分组，判断每个样本是单图，普通视频，还是导航视频
    # 压缩视觉 token 并投影到 Llama 的维度，最后生成供下游拼装使用的三份结果
    def vlm_attention(self, image_features, prompts=None, image_counts=None, long_video=False):
        # 读取压缩配置
        compress_type = self.config.compress_type
        compress_grid_sizes = {"grid:2": 4, "grid:4": 16, "mean": 1}

        # 检查压缩类型
        nav_size = compress_grid_sizes.get(compress_type)
        if nav_size is None:
            raise ValueError(f"Unsupported compress type: {compress_type}")

        if image_counts is None:
            assert len(image_features) == len(prompts), (
                f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
            )
        else:
            assert len(prompts) == len(image_counts), (
                f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
            )

        # 初始化输出容器和切片游标
        img_feat_lst = [] # 保存每个样本处理后的视觉 token
        video_or_not = [] # 保存每个样本后面应该走单图拼装分支还是视觉拼装分支
        nav_or_not = [] # 保存导航任务当前帧额外生成的 64 个高分辨率 token
        final_token_length_lst = [] # 一个死代码
        total_count = 0 # 对展平视觉特征进行连续切片的游标

        # 这部分是该函数的核心循环，每次循环处理一个视觉样本，而不是一张图片或者一帧
        '''
        1. 首先取出当前样本的全部帧
        2. 判断是不是导航任务
        3. 去掉 CLS token
        4. 压缩并投影视觉 token
        5. 把“帧维 × 每帧 token 维”合并
        6. 记录单图/视频/导航任务的标记
        7. 保存当前样本的视觉特征
        '''
        # 这里常常可能会有点看不懂，就是为什么是按照提示词字符串来进行遍历
        # 这里的关键就是 prompts 不是提示词字符串，而是每条样本的问题文本列表，它的外层维度恰好就是样本维度
        # 并且这里的 image feature 是已经被拍平了的 也就是没有样本的数量信息了
        for _idx, prompt in enumerate(prompts):
            assert isinstance(prompt, list), (
                f"Prompt should be a list, but got {type(prompt)}"
            )

            # 这里取出当前样本的视觉特征，判断它当前是不是导航样本并做硬校验
            # 确认帧数数值
            # 这里的 image feature 就是视觉 token
            # 这里的 image count 是指这个 batch 中各带多少帧图像
            # 这里跳的有点多，关于 image count 这个东西，首先这个东西如果在不为 None 的情况下就是一个列表
            
            # 这里的区分很关键，如果 image_counts 是 None，那么就说明这里 image feature 的 dim 0 是样本号
            # 那这里直接按照下标拿就可以了
            if image_counts is None:
                '''
                这里有一个很关键的一点就是，
                下面需要的 img_feat_prompt 这个变量需要是三维的
                但是 image_features[_idx] 这个变量是二维的
                所以这里需要在第二个维度上加一个维度，变成三维的，就是 image_features[_idx, None] 这个操作，补回一个长度为 1 的前置维度
                '''
                '''
                当然也可以使用 unsqueeze 来实现这个操作
                img_feat_prompt = image_features[_idx].unsqueeze(0)
                原本 squeeze() 的操作是挤掉一个长度为 1 的维度，unsqueeze() 的操作是指定位置补回一个长度为 1 的维度
                注意，这里长度为 1 不代表这个维度的这个唯一的 element 的值是 1，而是这个维度的长度是 1，里面的 element 的值可以是任意的，这里就是 image count 的值
                括号里面是 0 就是增加到第 0 维
                '''
                img_feat_prompt = image_features[_idx, None]
            # 如果 image_counts 不是 None，那么就说明这里 image feature 的 dim 0 是所有样本的帧数展平后的结果
            else:
                img_feat_prompt = image_features[total_count:total_count + image_counts[_idx]]
                total_count += image_counts[_idx]

            is_navigation = NAVIGATION_IDENTIFIER in prompt[0]
            # 判断是不是导航任务
            if is_navigation:
                if image_counts is None or image_counts[_idx] < 1 or len(prompt) != 1:
                    raise ValueError('[Navigation] wrong')

            # 去掉 CLS token，去掉视觉塔输出里排在最前面的 CLS token
            # 同时检查需要的图像特征和当前的 token 数
            # 如果不去掉 CLS token 下游就不能完全把 token 序列重新摊平成为二维网格然后后续做平均池化
            if (
                self.config.mm_vision_select_feature == 'patch'
                and img_feat_prompt.shape[1] % 2 == 1
            ):
                # 第一个 element 取全部维度，第二个 element 取从第 1 个到最后一个维度，这样就完成了切割
                img_feat_prompt = img_feat_prompt[:, 1:]

            # 把当前的样本的视觉 token 送去压缩和投影，换回两份可以直接用的 token
            '''
            final_token 主视觉 token：整段帧压缩后的结果：(帧数, 每帧 token 数, LLM 维度)
            final_token_nav 导航任务当前帧的高分辨率 token：导航样本额外的高分辨率当前帧 token：(1, 64, LLM 维度)
            '''
            final_token, final_token_nav = self.token_generation(
                img_feat_prompt,
                # 非常重要的数值，直接决定压缩模式
                image_counts=None if image_counts is None else image_counts[_idx],
                # 控制要不要额外生成一个 Nav token
                navigation=is_navigation,
            )

            if is_navigation and final_token_nav is None:
                raise ValueError('[Navigation] wrong')

            # 把输出展开到 4 维再合并到 3 维
            final_token = (
                final_token[None] # 在最前面加一个维度，变成 (1, 帧数, 每帧 token 数, LLM 维度)
                .expand(len(prompt), -1, -1, -1) # 这里把长度维 1 的维度撑开到指定的长度，-1 表示占位，表示该维度不变
                .flatten(1, 2) # 按照第 1 维和第 2 维展平，变成 (len(prompt), 帧数 * 每帧 token 数, LLM 维度)
            )

            # 看这一样本的视觉 token 到下游改用哪种拼法
            if image_counts is not None:
                if is_navigation:
                    final_token_nav = (
                        final_token_nav[None]
                        .expand(len(prompt), -1, -1, -1)
                        .flatten(1, 2)
                    )
                    assert (
                        final_token_nav.shape[0] == 1
                        and final_token_nav.shape[1] == 64
                        and final_token.shape[0] == 1
                    )
                    nav_or_not.append(final_token_nav)
                else:
                    nav_or_not.append(None)

                if image_counts[_idx] == 1:
                    if is_navigation:
                        assert final_token.shape[1] == nav_size
                        video_or_not.append(True)
                    else:
                        assert final_token.shape[1] == 64
                        video_or_not.append(False)
                else:
                    video_or_not.append(True)
            else:
                assert final_token.shape[1] == 64
                video_or_not.append(False)
                nav_or_not.append(None)

            img_feat_lst.append(final_token)

        # 最终返回这三个东西
        # 1. 投影到 LLM 维度的视觉 token 序列
        # 2. 每个样本是单图还是视频的标记
        # 3. 每个样本的导航任务当前帧的高分辨率 token（非导航样本为 None）
        return img_feat_lst, video_or_not, nav_or_not

    '''
    这个函数把帧当成批量维，一次性向量化处理所有的帧
    单帧在压缩前有着 256 个 patch token，这个 256 是 VIT 切出来的
    关键就是传进来的 vis_embed 这个变量，它的维度是 (帧数, 256, 1408)
    也就是说根据传进来的一个 3 维张量一个张量，也就是一个样本，可以包含多帧图像，每帧图像有 256 个 patch token，每个 patch token 的维度是 1408
    '''
    def token_generation(self, vis_embed, image_counts=None, navigation=False):
        # 内定义函数
        def process_grid(vis_embed, grid_size):
            # 这里开方反推正方形网格边长
            # ** 是开方运算，这里相当于开跟方
            cur_shape = int(vis_embed.shape[1] ** 0.5)
            assert grid_size > 1, f'Grid size should be larger than 1, but got {grid_size}'
            # 这里的 reshape 是把每帧的视觉 token 重新排列成一个正方形网格，方便后续做平均池化
            # 把一维的 token sequence reshape 成二维的正方形网格，方便后续做平均池化
            vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
            # 算出池化的块边长
            grid_stride = cur_shape // grid_size
            # 对空间网格做不重叠分块平均
            # 每个 grid_size x grid_size 的块被平均成一个 token，最终得到的视觉 token 数量就是 grid_size^2
            # 这个函数是 PyTorch 的二维平均池化函数，输入是一个四维张量，输出也是一个四维张量
            vis_embed = F.avg_pool2d(
                vis_embed.permute(0, 3, 1, 2),
                padding=0,
                kernel_size=grid_stride,
                stride=grid_stride,
            )
            # 把池化结果变换回最后一维是特征维度的形式，相当于是把两维合并成为一维
            return vis_embed.permute(0, 2, 3, 1).flatten(1, 2)

        grid_size = int(self.config.compress_type.split('grid:')[-1])
        # 这里做压缩逻辑的判断
        if image_counts is None or (image_counts == 1 and not navigation):
            vis_embed = process_grid(vis_embed, 8)
        # 导航任务情况下：
        # 最后一帧图像按照 grid 8 压缩成 64 个 token（高保真），其他的按照 grid size 粗压缩
        elif navigation:
            vis_embed_nav = vis_embed[-1:]
            vis_embed_nav = process_grid(vis_embed_nav, 8)
            vis_embed = process_grid(vis_embed, grid_size)
        else:
            vis_embed = process_grid(vis_embed, grid_size)

        # 投影池化后的视觉 token 到 Llama 的隐藏维度
        vis_embed = self.get_model().mm_projector(vis_embed)
        vis_embed_nav = self.get_model().mm_projector(vis_embed_nav) if navigation else None

        return vis_embed, vis_embed_nav

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    # 这个函数用来对进入 Llama transformer 的 input 进行处理
    # 处理文本 token + 图像特征 + 特殊图像 token 拼接成一个完整的输入序列，并且生成对应的 attention mask 和 labels
    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        prompts=None,
    ):
        # 根据 compress_type 来判断视觉 token 的压缩方式，并且计算每一帧压缩后应该占用多少个 token，结果保存在 nav_size 变量中
        # 主要是通过 grid 来进行控制
        # grid 决定了压缩网格的边长
        # nav_size 就是压缩后每一帧应该占用多少个 token
        """
        grid 越大，压缩后的空间网格越细
        每张图片保留的视觉 token 越多，空间细节越完整
        """
        if 'grid' in self.config.compress_type:
            grid_size = int(self.config.compress_type.split('grid:')[-1])
            if grid_size == 2:
                nav_size = 4
            elif grid_size == 4:
                nav_size = 16
            else:
                raise ValueError
        elif 'mean' in self.config.compress_type:
            nav_size = 1
        else:
            raise ValueError

        # 做一个 prompt 的兜底
        # 如果这次调用没有传入 prompt，就尝试使用之前缓存到模型对象里的 self.prompt
        if prompts is None and hasattr(self, 'prompts'):
            prompts = self.prompts

        # 取得视觉编码器对象
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (
                        attention_mask.shape[0],
                        past_key_values[-1][-1].shape[-2] + 1,
                    ),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        # 取 batch 里的第一条样本，然后看其的最后一维的大小，超过 1000 就认为是预计算好的特征，而不是像素图
        if images[0].shape[-1] > 1000:
            long_video = True
        else:
            long_video = False

        # 把 image 的两种输入形态归一化，并交给视觉编码器，整体就四步
        # 首先先判断是不是 list 或者是 5 维的 tensor
        if type(images) is list or images.ndim == 5:
            # 把 list 中的每一个元素统一成 4 维的 tensor
            # 不然下面的拼接就会有问题
            if not long_video:
                images = [
                    image if len(image.shape) == 4 else image.unsqueeze(0)
                    for image in images
                ]
            # 记下每个样本各有多少帧，得到一个列表
            # 这里这个 image count 变量会被一直传递到后边的 vlm_attention 函数中去使用，期间不会再被改变
            # 这个列表的长度就是样本的数量
            # 这个列表的每个 element 的数值就是这个样本有多少个图像
            # 总样本数就是一整个 batch size
            image_counts = [image.shape[0] for image in images]
            # 把所有样本沿着第 0 维拼成一整块
            # 这里是沿着第 0 维拼接的，因为第 0 维是 batch 维度
            concat_images = torch.cat(images, dim=0)
            # 调用观测编码入口
            # 拿到返回的编码后的视觉 token，每个样本内部到底是按视频处理还是按单图处理的标记，每个导航样本额外的高分辨率当前观测 token（非导航样本为 None）
            image_features, video_or_not, nav_or_not = self.encode_images(
                concat_images,
                prompts,
                image_counts,
                long_video=long_video,
            )
        # 5 维 tensor 直接送进去编码器
        else:
            image_features, video_or_not, nav_or_not = self.encode_images(
                images,
                prompts,
                long_video=long_video,
            )

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat(
                    [
                        cur_input_embeds_1,
                        cur_image_features[0:0],
                        cur_input_embeds_2,
                    ],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if not long_video:
                token_idx = 0
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx][token_idx]
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0]

                    if (
                        getattr(self.config, 'tune_mm_mlp_adapter', False)
                        and getattr(self.config, 'mm_use_im_start_end', False)
                    ):
                        raise ValueError('wrong')
                        cur_new_input_embeds.append(
                            self.get_model()
                            .embed_tokens(cur_input_ids[:image_token_start - 1])
                            .detach()
                        )
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1:image_token_start]
                            )
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[image_token_start + 1:image_token_start + 2]
                            )
                        )
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                            cur_new_labels.append(
                                cur_labels[image_token_start:image_token_start + 1]
                            )
                            cur_labels = cur_labels[image_token_start + 2:]
                    else:
                        if (
                            nav_or_not[cur_image_idx] is None
                            and video_or_not[cur_image_idx] is False
                        ):
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[:image_token_start]
                                )
                            )
                            cur_new_input_embeds.append(cur_image_features)
                            assert cur_image_features.shape[0] == 64

                        elif (
                            nav_or_not[cur_image_idx] is None
                            and video_or_not[cur_image_idx] is True
                        ):
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[:image_token_start]
                                )
                            )
                            seperator_token = self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1, None]
                            )
                            video_index = 0
                            assert len(cur_image_features) % nav_size == 0

                            for ii in range(int(len(cur_image_features) / nav_size)):
                                cur_new_input_embeds.append(
                                    cur_image_features[video_index:video_index + nav_size]
                                )
                                if ii == (len(cur_image_features) / nav_size) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += nav_size
                        else:
                            assert video_or_not[cur_image_idx] is True
                            assert token_idx == 0
                            assert nav_or_not[cur_image_idx][token_idx].shape[0] == 64
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[:image_token_start]
                                )
                            )
                            seperator_token = self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1, None]
                            )
                            video_index = 0
                            assert len(cur_image_features) % nav_size == 0
                            for ii in range(int(len(cur_image_features) / nav_size)):
                                cur_new_input_embeds.append(
                                    cur_image_features[video_index:video_index + nav_size]
                                )
                                if ii == (len(cur_image_features) / nav_size) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += nav_size
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[image_token_start + 1:image_token_start + 3]
                                )
                            )
                            cur_new_input_embeds.append(nav_or_not[cur_image_idx][token_idx])

                        if labels is not None:
                            if (
                                nav_or_not[cur_image_idx] is None
                                and video_or_not[cur_image_idx] is False
                            ):
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_labels = cur_labels[image_token_start + 1:]
                            elif (
                                nav_or_not[cur_image_idx] is None
                                and video_or_not[cur_image_idx] is True
                            ):
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_new_labels.append(
                                    torch.full(
                                        (int(cur_image_features.shape[0] / nav_size - 1),),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_labels = cur_labels[image_token_start + 1:]
                            else:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_new_labels.append(
                                    torch.full(
                                        (int(cur_image_features.shape[0] / nav_size - 1),),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_new_labels.append(
                                    torch.full(
                                        (nav_or_not[cur_image_idx][token_idx].shape[0] + 2,),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_labels = cur_labels[image_token_start + 3:]

                    if (
                        getattr(self.config, 'tune_mm_mlp_adapter', False)
                        and getattr(self.config, 'mm_use_im_start_end', False)
                    ):
                        raise ValueError('wrong')
                    else:
                        if nav_or_not[cur_image_idx] is not None:
                            cur_input_ids = cur_input_ids[image_token_start + 3:]
                        else:
                            cur_input_ids = cur_input_ids[image_token_start + 1:]
                    image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                    token_idx += 1

                # changle image idx after processing one sample
                cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    if (
                        getattr(self.config, 'tune_mm_mlp_adapter', False)
                        and getattr(self.config, 'mm_use_im_start_end', False)
                    ):
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids).detach()
                        )
                    else:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids)
                        )
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    assert cur_new_input_embeds.shape[0] == cur_new_labels.shape[0]
                    new_labels.append(cur_new_labels)
            else:
                cur_new_input_embeds = torch.Tensor(
                    len(cur_input_ids),
                    self.config.hidden_size,
                ).to(dtype=self.dtype, device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if (
                    not self.training
                    and self.get_model().embed_tokens.weight.device != cur_input_ids.device
                ):
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(device=model_device)
                    cur_new_input_embeds[text_token_indices] = (
                        self.get_model()
                        .embed_tokens(cur_input_ids_text)
                        .to(device=data_device)
                    )
                else:
                    cur_new_input_embeds[text_token_indices] = self.get_model().embed_tokens(
                        cur_input_ids[text_token_indices]
                    )
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (
                                max_len - cur_new_embed.shape[0],
                                cur_new_embed.shape[1],
                            ),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for (
                    cur_attention_mask,
                    cur_new_labels,
                    cur_new_labels_align,
                ) in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask),
                    dim=1,
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # 返回结果
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        tokenizer.add_tokens(
            [
                VIDEO_START_SPECIAL_TOKEN,
                VIDEO_END_SPECIAL_TOKEN,
                IMAGE_START_TOKEN,
                IMAGE_END_TOKEN,
                NAVIGATION_SPECIAL_TOKEN,
                IAMGE_SEPARATOR,
            ],
            special_tokens=True,
        )
        self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter,
                    map_location='cpu',
                )
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        "Unexpected embed_tokens_weight shape. "
                        f"Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. "
                        f"Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
