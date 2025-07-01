from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "./final_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"🤖 GPT-2: {generated_text.strip()}")