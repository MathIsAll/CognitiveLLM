import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import time
from tqdm import tqdm
import csv
from transformers import LlamaModel, LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model


def format_instruction(sample):
    # data_output = sample['text']
    data_input = sample['text']  # + "\n" + sample['input'] + "\n"
    return data_input


def tokenize_function(examples):
    i = format_instruction(examples)
    input_ids = tokenizer(i, padding="max_length", truncation=True, max_length=2048)["input_ids"]
    return {"input_ids": input_ids}


def data_collator(features):
    # 将所有的features的input_ids和labels分别堆叠起来
    input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}

# 获取随机元素位置
def get_random_indices_percentage(matrix, percentage=0.1):
    """
    获取矩阵中随机percentage比例元素的位置

    参数:
        matrix: 输入张量
        percentage: 要抽取的元素比例(0-1之间)

    返回:
        indices: 随机元素的索引张量
    """
    total_elements = matrix.numel()
    k = max(1, int(total_elements * percentage))  # 至少取1个元素

    # 生成随机排列并取前k个
    indices_1d = torch.randperm(total_elements)[:k]

    # 将一维索引转换为多维索引
    indices_multidim = torch.stack(torch.unravel_index(indices_1d, matrix.shape), dim=1)

    return indices_multidim

def get_grad(model, grad_dict, find_large_index=True):
    for name, param in model.named_parameters():
        # 忽略embed层和最合lm_head的参数
        if 'embed_tokens' in name or 'lm_head' in name:
            continue

        weight_grad = param.grad
        if name not in grad_dict:
            grad_dict[name] = torch.zeros_like(weight_grad.view(-1), device='cuda:0', dtype=torch.long)
            # ones_dict[name] = torch.ones(int(0.05 * weight_grad.numel()), device='cuda:0', dtype=torch.long)

        # 找到最大或最小的前20%的元素的索引
        _, indices = torch.topk(torch.abs(weight_grad).view(-1), int(0.1 * weight_grad.numel()), largest=find_large_index)

        # 把grad_dict[name]中对应的索引位置加上对应的次数
        record_gpu = grad_dict[name]
        record_gpu.index_add_(0, indices.to('cuda:0'), torch.ones(int(0.1 * weight_grad.numel()), device='cuda:0', dtype=torch.long))

        # grad_dict[name] = record_gpu.to('cpu')


# Hugging Face model id
model_id = "/data/jshr/pretrained_models/llama2-7b-hf"  # non-gated

# 加载模型
######################
model_path = "/data/jshr/pretrained_models/llama2-7b-hf"
cuda_list = '1'.split(',')
memory = '80GiB'
no_split_module_classes = LlamaForCausalLM._no_split_modules

max_memory = {int(cuda): memory for cuda in cuda_list}
config = LlamaConfig.from_pretrained(model_path)
with init_empty_weights():
    model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)  # 加载到meta设备中，不需要耗时，不需要消耗内存和显存

device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)  # 自动划分每个层的设备
load_checkpoint_in_model(model, model_path, device_map=device_map)  # 加载权重
model = dispatch_model(model, device_map=device_map)  # 并分配到具体的设备上
#######################

model.train()

tokenizer = AutoTokenizer.from_pretrained("/data/jshr/pretrained_models/llama2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def get_and_save_grad(file_type):
    # 加载数据集
    dataset = load_dataset('text', data_files=f'/data/jshr/pretraining_data/baai_industrycorpus_{file_type}_mrsn_find.txt', split='train')
    tokenized_datasets = dataset.map(tokenize_function)
    print("数据集规模:", len(tokenized_datasets))
    print("数据集示例:", format_instruction(dataset[0]))

    data_loader = Data.DataLoader(tokenized_datasets, batch_size=2, shuffle=True, collate_fn=data_collator)

    grad_dict = {}

    time_start = time.time()
    for epoh in range(1):
        for step, batch in tqdm(enumerate(data_loader)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            get_grad(model, grad_dict, find_large_index=True)
            model.zero_grad()
            if step > 20000:
                break

        print('time:', time.time() - time_start)

    save_dict = {}

    for name in tqdm(grad_dict):
        test = grad_dict[name]

        # 找出tensor中最大的前20%的元素的索引
        indices = torch.topk(test, int(0.1 * test.numel()))[1].to('cpu')

        save_dict[name] = indices

    torch.save(save_dict, f"/data/jshr/pretraining_data/finded_index/highest_index_output_{file_type}_subnetwork.pt")


for domain in ["medicine", "finance", "law", "computer"]:
    get_and_save_grad(domain)
