# 实现20%微调的训练
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.optim
from datasets import load_dataset, config, load_from_disk
from pathlib import Path

os.environ["WANDB_MODE"] = "offline"

# local_rank = int(os.environ["LOCAL_RANK"])  # torchrun 自动设置
# device = torch.device(f"cuda:{local_rank}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = ["Factual_Knowledge", "Procedural_Knowledge", "Critical_Thinking", "Conceptual_Reasoning", "Causal_Inference", "Generative_Thinking"]


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


# 循环处理每个任务
for grad_strategy in ["min", "random"]:
    for task_id, current_task in enumerate(task_name):
        print(f"正在处理任务 {task_id}: {current_task}")

        # 设置模型路径：第一个任务使用基础模型，后续任务使用前一个任务的检查点
        if task_id == 0:
            model_path = "/HOME/hitsz_qcchen/hitsz_qcchen_1/HDD_POOL/Qwen2.5-1.5B-base"
        else:
            # 使用前一个任务的检查点
            prev_task = task_name[task_id - 1]
            model_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/{prev_task}_min"

        print(f"使用模型路径: {model_path}")

        # 指定包含所有JSON文件的目录路径
        folder_path = f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/ds_distilled_samples/distill_{current_task}.json"

        dataset = load_dataset('json', data_files=folder_path, split='train', streaming=True)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        train_dataset = dataset.map(tokenize_function)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # 加载索引
        index = torch.load(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/SelectedParams/{current_task}_{grad_strategy}_qwen2.5-1.5b.pt", map_location='cpu', weights_only=True)
        param_to_index = {}  # 40G
        for name, param in model.named_parameters():
            if 'embed_tokens' in name or 'lm_head' in name:
                continue
            param_to_index[param] = index[name].to(param.device)

        trainer_args = TrainingArguments(
            output_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/{current_task}_min",
            num_train_epochs=1,
            per_device_train_batch_size=3,
            save_steps=2000,
            save_total_limit=5,
            prediction_loss_only=True,
            logging_dir=f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/logs/{current_task}_min",
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
            param_to_index=param_to_index
        )

        trainer.train()

        model.save_pretrained(f"/HOME/hitsz_qcchen/hitsz_qcchen_1/new_disk/SPUpdater-main/fined_chkpts/{grad_strategy}_{current_task}")

        print(f"策略 {grad_strategy} 任务 {current_task} 处理完成")

print("所有任务处理完成！")