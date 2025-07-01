from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Tokenizer
from datasets import load_from_disk

train_dataset = load_from_disk("tokenized_combined_dialog_data/train")
val_dataset = load_from_disk("tokenized_combined_dialog_data/val")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id

training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=5000,
    save_steps=10000,
    save_total_limit=3,
    logging_steps=100,
    eval_strategy="steps",
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    fp16=True,
    logging_dir="./logs",
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")