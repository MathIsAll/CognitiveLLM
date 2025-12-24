# 实现20%微调的训练（EVA 初始化版）
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig, get_peft_model, TaskType,  # CHANGED: 使用 get_peft_model
    EvaConfig, initialize_lora_eva_weights, # CHANGED: 引入 EVA
    PeftModel
)
import torch
import torch.optim
from datasets import load_dataset, config, load_from_disk
from torch.utils.data import DataLoader        # CHANGED: 为 EVA 初始化准备
from pathlib import Path
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["WANDB_MODE"] = "offline"

local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 自动设置
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
world_size = dist.get_world_size()

task_name = ["Factual_Knowledge", "Procedural_Knowledge", "Critical_Thinking", "Conceptual_Reasoning", "Causal_Inference", "Generative_Thinking", "Code"]
task_id = 6

def format_instruction(sample):
    data_input = sample['instruct'] + "\n" + sample['input'] + '\n' + sample['output']
    return data_input.strip()

def tokenize_function(examples):
    text = format_instruction(examples)
    input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=4096)["input_ids"]
    return {"input_ids": input_ids}

def data_collator(features):
    input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}

model_path = "/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/Qwen2.5-1.5B-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 假设该磁盘数据集已包含 input_ids（与你原脚本一致）
train_dataset = load_from_disk(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/ds_distilled_instructs/instruct_{task_name[task_id]}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

# =========================
# CHANGED: 配置 LoRA + EVA
# =========================
"""
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 你原先的目标层
    lora_dropout=0.05,
    bias="none",
    init_lora_weights="eva",                # 关键：启用 EVA 初始化
    eva_config=EvaConfig(                   # 关键：传入 EvaConfig
        rho=2.0,                            # 允许自适应扩大总秩的上限系数（常用 1.5~3
    )
)

# 将 LoRA 注入到模型
model = get_peft_model(model, lora_config, low_cpu_mem_usage=True)

# ==========================================
# CHANGED: 用你自己的数据执行一次 EVA 初始化
# ==========================================
# 说明：EVA 仅需前向传播，不计算梯度。这里复用你的 data_collator。
#      只取少量 batch（由 EvaConfig.svd_sample_batches 控制）即可完成初始化。
# Create sampler for distributed training
sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
evaluator_loader = DataLoader(
    train_dataset,
    batch_size=2,               # 要与 eva_config.svd_batch_size 保持一致
    shuffle=False,
    collate_fn=data_collator,
    sampler=sampler
)

model.eval()
torch.set_grad_enabled(False)
initialize_lora_eva_weights(model, evaluator_loader)  # 核心初始化步骤
torch.set_grad_enabled(True)
model.train()
"""
peft_model_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/peft_chkpts/eva_peft/{task_name[task_id-1]}"
model = PeftModel.from_pretrained(model, peft_model_path, is_trainable=True)

trainer_args = TrainingArguments(
    output_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/peft_chkpts/eva_peft/{task_name[task_id]}",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=2000,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/logs/{task_name[task_id]}",
    logging_steps=10,
    logging_first_step=True,
    optim='adamw_torch',
    learning_rate=1e-5,
    warmup_steps=0,
    lr_scheduler_type='cosine',
    gradient_accumulation_steps=8,
    # gradient_checkpointing=True,
    max_grad_norm=1,
    adam_epsilon=1e-5,
    bf16=True,
    resume_from_checkpoint=True
)

trainer = Trainer(
    model=model,
    args=trainer_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/peft_chkpts/eva_peft/{task_name[task_id]}")