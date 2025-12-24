import os
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.optimizers import create_lorafa_optimizer
from lomo_optim import AdaLomo  # 导入 AdaLomo 优化器

os.environ["WANDB_MODE"] = "offline"

local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 自动设置
device = torch.device(f"cuda:{local_rank}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = ["Factual_Knowledge", "Procedural_Knowledge", "Critical_Thinking", "Conceptual_Reasoning", "Causal_Inference", "Generative_Thinking", "Code"]

task_id = 6


# 构建Qwen2.5指令模板[1,6](@ref)
PROMPT_TEMPLATE = """
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}
"""

# 关键修改1：简化tokenization函数
def tokenize_function(examples):
    # 构建完整指令文本
    text = []
    for id in range(len(examples["instruct"])):
        text += [PROMPT_TEMPLATE.format(
            system_message="你是一个有帮助的助手",
            instruction=examples["instruct"][id],
            response=examples["output"][id] + tokenizer.eos_token)  # 添加EOS标记
        ]
    # text = examples['instruct'] + "\n" + examples['input'] + '\n' + examples['output']
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=4096,
        return_tensors="pt"  # 确保返回PyTorch张量
    )


# 加载数据集
# folder_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/ds_distilled_samples/distill_{task_name[task_id]}.json"
# dataset = load_dataset('json', data_files=folder_path, split='train', streaming=True)


# 加载模型和分词器
model_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/adalomo/{task_name[task_id-1]}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 确保decoder-only模型使用右侧填充

# 关键修改2：使用DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 设置为False用于因果语言建模
    pad_to_multiple_of=8  # 可选，提高计算效率
)

# 处理数据集
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = load_from_disk(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/ds_distilled_instructs/instruct_{task_name[task_id]}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

# 训练参数配置
trainer_args = TrainingArguments(
    output_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/adalomo/{task_name[task_id]}",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=2000,
    save_total_limit=5,
    prediction_loss_only=True,
    logging_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/logs/{task_name[task_id]}",
    logging_steps=10,
    logging_first_step=True,
    optim='adalomo',
    learning_rate=1e-5,
    warmup_steps=0,
    lr_scheduler_type='cosine',
    gradient_accumulation_steps=8,
    max_grad_norm=1,
    adam_epsilon=1e-5,
    bf16=True,
    resume_from_checkpoint=True
)

# 关键修改3：使用新的数据整理器
trainer = Trainer(
    model=model,
    args=trainer_args,
    data_collator=data_collator,  # 使用标准整理器
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
# trainer.train(resume_from_checkpoint=f"/PATH/checkpoint-8000")

# 保存模型
model.save_pretrained(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/adalomo/{task_name[task_id]}")
