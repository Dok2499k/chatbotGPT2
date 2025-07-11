from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
from evaluate import load
from tqdm import tqdm
import torch

model_path = "./final_model"
val_data_path = "tokenized_combined_dialog_data/val"

tokenizer = GPT2Tokenizer.from_pretrained("./model_output")
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

val_dataset = load_from_disk(val_data_path)

bleu = load("bleu")
predictions, references = [], []

print("Evaluating the model...")
for i in tqdm(range(len(val_dataset))):
    item = val_dataset[i]

    input_ids = torch.tensor([item["input_ids"]], device=device)
    attn_mask = torch.tensor([item["attention_mask"]], device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    decoded_output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    decoded_target = tokenizer.decode([id for id in item["labels"] if id != -100], skip_special_tokens=True)

    predictions.append(decoded_output.strip())
    references.append([decoded_target.strip()])

if all(pred.strip() == "" for pred in predictions):
    print("All predictions were empty. BLEU could not be calculated.")
else:
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print(f"\n✅ BLEU Score: {bleu_score['bleu']:.4f}")
