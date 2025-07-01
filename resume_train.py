from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch

train_dataset = load_from_disk("tokenized_combined_dialog_data/train")
val_dataset = load_from_disk("tokenized_combined_dialog_data/val")

checkpoint_path = "./model_output/checkpoint-100000"
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
tokenizer = GPT2Tokenizer.from_pretrained("./model_output")


training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=False,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=5000,
    save_steps=10000,
    save_total_limit=3,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=100,
    logging_first_step=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train(resume_from_checkpoint=checkpoint_path)


trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")