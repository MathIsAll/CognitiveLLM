# 实现20%微调的训练
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import BoneConfig, get_peft_model, PeftModel
import torch
import torch.optim
from datasets import load_dataset, config, load_from_disk
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"

local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 自动设置
device = torch.device(f"cuda:{local_rank}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = ["Factual_Knowledge", "Procedural_Knowledge", "Critical_Thinking", "Conceptual_Reasoning", "Causal_Inference", "Generative_Thinking", "Code"]
task_id = 6

def format_instruction(sample):
    # data_output = sample['text']

    data_input = sample['instruct'] + "\n" + sample['input'] + '\n' + sample['output']
    return data_input.strip()


def tokenize_function(examples):
    text = format_instruction(examples)
    input_ids = tokenizer(text, padding="max_length", truncation=True, max_length=4096)["input_ids"]

    return {"input_ids": input_ids}


def data_collator(features):
    # 将所有的features的input_ids和labels分别堆叠起来
    input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
    labels = input_ids.clone()

    return {"input_ids": input_ids, "labels": labels}

# 指定包含所有JSON文件的目录路径
"""
folder_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/deepseek-distill-ds/Factual_Knowledge/{task_name[task_id]}"

# 获取目录下所有JSON文件的完整路径（支持递归搜索子目录）
json_files = [
    os.path.join(folder_path, f) 
    for f in os.listdir(folder_path) 
    if f.endswith('.json')
]
print(json_files)
"""
# dataset = load_dataset('json', data_files=json_files, split='train', streaming=True)

model_path = "/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/Qwen2.5-1.5B-base" # f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/full_parameter/{task_name[task_id-1]}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# train_dataset = dataset.map(tokenize_function)
train_dataset = load_from_disk(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/ds_distilled_instructs/instruct_{task_name[task_id]}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

"""
bone_config = BoneConfig(
    task_type="CAUSAL_LM",  # 因果语言模型任务
    r=16,  # LoRA秩，控制低秩矩阵的维度[3,5](@ref)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 针对注意力机制的关键层[5](@ref)
    init_weights=True,
    bias="none",  # 不调整偏置参数[3](@ref)
)

# 应用LoRA到模型[3,4](@ref)
model = get_peft_model(model, bone_config)
"""
peft_model_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/peft_chkpts/bone_peft/{task_name[task_id-1]}"
model = PeftModel.from_pretrained(model, peft_model_path, is_trainable=True)

trainer_args = TrainingArguments(
    output_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/bone_peft/{task_name[task_id]}",
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
    # gradient_checkpointing= True,
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
# trainer.train(resume_from_checkpoint=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/{task_name[0]}/checkpoint-8000")

model.save_pretrained(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/SPUpdater-main/fined_chkpts/bone_peft/{task_name[task_id]}")
