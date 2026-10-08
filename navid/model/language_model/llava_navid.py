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


from typing import List, Optional, Tuple, Union

# 导入 pytorch 底层
import torch

# torch.nn 是 pytorch 中专门用来构建神经网络的模块，里面提供了大量常用的网络组件
"""
比如：
nn.Module          # 所有神经网络模块的基础类
nn.Linear          # 全连接层
nn.Conv2d          # 二维卷积层
nn.Embedding       # 词向量/Embedding 层
nn.LayerNorm       # Layer Normalization
nn.ReLU            # ReLU 激活函数
nn.Dropout         # Dropout
nn.Softmax         # Softmax
nn.CrossEntropyLoss # 交叉熵损失函数
"""

import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 导入 transformers 库
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# 导入 navid 库，其中包含自己实现的类
from navid.model.navid_arch import NaVidMetaModel, NaVidMetaForCausalLM
from navid.constants import NAVIGATION_IDENTIFIER


# 完全继承 Llava 的类属性，但是改一个字段
class LlavaConfig(LlamaConfig):
    model_type = "llava"


# 这个类同时继承了开源的 LlamaForCausalLM 和自己实现的 NaVidMetaForCausalLM
# 这样就使得这个模型同时具有了两个父类的能力
# 实际的基座模型还是 llama 而不是 llava
"""
这里的 llava 主要是指一个视觉语言架构的名字
全程 large language and vision assistant
官方的 llava 就是把 视觉编码器 + 视觉特征投影层 + llama vicuna 这种语言模型 这三个东西组合在一起
"""
class LlavaAttLlamaModel(NaVidMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAttLlamaModel, self).__init__(config)


# 把“多模态输入拼装”和“Llama 文本生成”串起来形成一个完整的前向传播
class LlavaLlamaAttForCausalLM(LlamaForCausalLM, NaVidMetaForCausalLM):
    # 告诉 transformer 这个类的 config 是 LlavaConfig
    config_class = LlavaConfig

    # python 里面的构造函数
    # 外部执行调用
    def __init__(self, config):
        # 这里跳过了具体的初始化过程，直接调用子类的构造函数
        # 跳过了 LlamaForCausalLM
        super(LlamaForCausalLM, self).__init__(config)

        # 这里创建真正的 transformer 主体，并且保存到当前模型的 self.model 里面
        # 其同时继承了 NaVidMetaModel 和 LlamaModel
        # 同时拥有了 Llama 的 transformer 文本建模能力和 NaVid 的多模态输入的视觉编码器和投影层能力
        # 这个负责理解输入并且产生隐藏状态
        self.model = LlavaAttLlamaModel(config)

        # 创建模型输出层
        # nn.linear 是一个全连接层
        # 这里传入了三个参数，输入维度，输出维度，不使用偏置（False）
        # 其会将每个位置上的 N 维 vector 映射成为 M 个分数，每个分数对应词表中的一个 token 的概率
        # 这个负责把隐藏状态转换成词表上的预测分数
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重
        # 一般在所有的子模块都创建完成之后执行，用于完成模型的权重初始化
        """
        一般包括：
        1. 根据模型规则初始化权重
        2. 初始化线性层，Embedding，LayerNorm 蹬模块
        3. 处理权重绑定
        4. 等等
        """
        # 一般都要先创建完 model 再创建完 head 最后初始化，这样初始化才能够看到完整的模型结构
        self.post_init()

    # 返回模型
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    # 返回值类型预计
    # Union 表示多个可能的类型，前面有导入
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 配置三个可选控制参数
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # 检查输入数据和模型是否位于同一设备上
        # 应该训练模式也要检查
        if not self.training:
            # 检查图像是否处于同一设备上，如果不是，则将其移动到模型所在的设备上
            if images[0].device != self.device:
                images[0] = images[0].to(device=self.device)
            # 检查输入的 token 是否处于同一设备上，如果不是，则将其移动到模型所在的设备上
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)

        # 这部分代码是在进入 Llamam transformer 之前对文本，图像和标签进行多模态输入整理
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        # 调用的是 navid_arch.py 里面的 prepare_inputs_labels_for_multimodal 函数
        # 这个函数的核心任务是把文本 token + 图像特征 + 特殊图像 token 拼接成一个完整的输入序列，并且生成对应的 attention mask 和 labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            prompts=prompts,
        )

        # 释放 PyTorch 的 CUDA 缓存分配器中当前没有被 Tensor 使用的显存缓存
        # PyTorch 中大致分为两种显存
        # 1. 正在被 Tensor 占用的显存
        # 2. PyTorch 暂时缓存，但是当前没有被使用的显存
        torch.cuda.empty_cache()

        # 调用多模态 Llama Transformer 的主体模型，让其根据已经准备好的文本和视觉 embedding 来生成隐藏状态
        # 这里 slef.model 之所以可以像一个函数一样被调用的前提是其是一个 PyTorch的 nn.Module 对象，并且实现了 forward 方法
        # 其实际上会触发 model._call_() 方法，进而调用 model.forward() 方法
        """
        在 Python 中如果一个对象实现了 __call__ 方法，那么这个对象就可以像函数一样被调用
        """
        # 这里 LlavaAttLlamaModel 没有写 _call_ 方法，所以 python 会继续沿着父类继承往上找，直到找到最终的 nn.Module 有 _call_ 方法，才调用其
        # 同理，想要调用 forward 方法，python 会继续沿着父类继承往上找，直到找到最终的 LlamaModel 有 forward 方法，才调用其
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 按照逻辑选择最后一层全部的隐藏状态
        hidden_states = outputs[0]
        # 这里的 lm_head 是一个全连接层，输入是隐藏状态，输出是词表上每个 token 的预测分数
        # 这里输出的是未归一化的输出分数
        logits = self.lm_head(hidden_states)

        # 这部分代码负责计算训练损失 loss
        # 这里先初始化 loss
        loss = None
        # 这里通过 label 来判断当前是否是训练模式，如果是训练模式就计算 loss
        if labels is not None:
            # 这里就是在实现训练里面的错位 label
            # 先取 logits 的前 n-1 个 token 的预测分数
            shift_logits = logits[..., :-1, :].contiguous()
            # 再取 labels 的后 n-1 个 token 的真实标签
            # 这样每一位的预测分数就对应了下一位的真实标签
            shift_labels = labels[..., 1:].contiguous()
            # 创建一个交叉熵损失函数对象
            loss_fct = CrossEntropyLoss()
            # 把之前的 shift_logits 和 shift_labels 拉平为二维和一维
            # 这里主要是因为 CrossEntropyLoss 的输入要求是二维的预测分数和一维的真实标签
            # 这里输入 -1 让 PyTorch 自动计算维度大小，保证总元素数量不变
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # 同理，label 也要被拉平为一维，这样和之前的 shift_logits 维度对应起来
            shift_labels = shift_labels.view(-1)
            # 把标签移动到和预测分数相同的设备上，保证计算 loss 的时候不会出现设备不一致的错误
            shift_labels = shift_labels.to(shift_logits.device)
            # 最后计算交叉熵损失，得到每个 token 的预测分数和真实标签之间的差异
            # 然后汇总成为一个平均 loss
            loss = loss_fct(shift_logits, shift_labels)

        # 根据 return_dict 参数来决定返回值的格式
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 这个函数在每一步生成新的 token 时被调用，用于准备输入数据
    # 这个函数在 forward 函数被调用之前就被调用了，用于准备输入数据
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)
