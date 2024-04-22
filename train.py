# original script: https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py
import torch
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from transformers import DataCollatorForLanguageModeling
import time
import sys

import transformer_engine.common.recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast, fp8_model_init

assert sys.argv[1] in ("bf16", "fp8"), "provide correct mixed precision type!"

mixed_precision = sys.argv[1]
model_name = "meta-llama/Meta-Llama-3-8B"
dataset_name = "timdettmers/openassistant-guanaco"
dataset_text_field = "text"
learning_rate = 1.41e-5
batch_size = int(sys.argv[2])
max_seq_length = 256
gradient_accumulation_steps = 1
peft_lora_r = 64
peft_lora_alpha = 16
num_training_steps=int(sys.argv[3])

config = AutoConfig.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
)

model.config.use_cache=False

def get_dataloaders(accelerator:Accelerator, batch_size:int = 8):
    dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    with accelerator.main_process_first():
        dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )

    pad_to_multiple_of = 16
    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader

accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision)
accelerator.print(f'State: {accelerator.state}')
train_dataloader = get_dataloaders(accelerator, batch_size)

optimizer = AdamW(params = model.parameters(), lr=learning_rate)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
# print(f"trainable params: {model.print_trainable_parameters()}")

accelerator.init_trackers("fp8-benchmarks", config={
    "model_name": model_name,
    "dataset_name": dataset_name,
    "batch_size": batch_size,
    "accelerator_state": accelerator.state,
    "mixed_precision": accelerator.mixed_precision,
},
init_kwargs={"wandb": {"name": f'{accelerator.mixed_precision}_bs_{batch_size}_{accelerator.num_processes}_gpus'}})

model.train()
completed_steps = 0
total_loss = 0
optimizer.zero_grad()
# 100 just to get the full time in
for _ in range(100):
    if completed_steps >= num_training_steps:
        break
    for step, batch in enumerate(train_dataloader):
        start_time = time.time()
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            print(f"Step {step}: loss: {loss.item()}, batch shape: {batch['input_ids'].shape}, peak gpu mem: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            completed_steps += 1

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Step {step} time {total_time}")
        accelerator.log({"batch_time": total_time, "input_ids": batch["input_ids"].cpu().numpy(), "attention_mask": batch["attention_mask"].cpu().numpy()})
        start_time = end_time

        if completed_steps >= num_training_steps:
            break

accelerator.end_training()
