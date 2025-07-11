from convokit import Corpus, download
from transformers import GPT2Tokenizer
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets

corpus = Corpus(filename=download("movie-corpus"))

inputs_cornell, targets_cornell = [], []

for convo_id in tqdm(corpus.get_conversation_ids(), desc="Processing Cornell"):
    convo = corpus.get_conversation(convo_id)
    utterances = [corpus.get_utterance(uid) for uid in convo.get_utterance_ids()]
    if len(utterances) < 2:
        continue

    for i in range(len(utterances) - 1):
        in_text = utterances[i].text.strip()
        out_text = utterances[i + 1].text.strip()
        if in_text and out_text:
            inputs_cornell.append(in_text)
            targets_cornell.append(out_text)

cornell_dataset = Dataset.from_dict({
    "input_text": inputs_cornell,
    "target_text": targets_cornell
})

persona = load_dataset("bavard/personachat_truecased")

def preprocess_persona(example):
    if example["history"] and len(example["history"]) >= 2:
        in_text = example["history"][-2].strip()
        out_text = example["history"][-1].strip()
        return {"input_text": in_text, "target_text": out_text}
    return {"input_text": "", "target_text": ""}

persona_processed = persona["train"].map(
    preprocess_persona, batched=False,
    remove_columns=persona["train"].column_names,
    desc="Processing Persona"
)

persona_processed = persona_processed.filter(
    lambda x: x["input_text"] and x["target_text"]
)

combined_dataset = concatenate_datasets([cornell_dataset, persona_processed])
train_test = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, val_dataset = train_test["train"], train_test["test"]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_pairs(example):
    input = example["input_text"]
    target = example["target_text"]

    max_len = 256
    max_input_len = 128
    max_target_len = max_len - max_input_len - 2

    input_ids = tokenizer.encode(input, add_special_tokens=False)[:max_input_len]
    target_ids = tokenizer.encode(target, add_special_tokens=False)[:max_target_len]

    input_ids_combined = input_ids + [tokenizer.eos_token_id] + target_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids_combined)

    pad_len = max_len - len(input_ids_combined)
    input_ids_combined += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len

    labels = [-100] * (len(input_ids) + 1) + target_ids + [tokenizer.eos_token_id]
    labels += [-100] * pad_len

    return {
        "input_ids": input_ids_combined,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_train = train_dataset.map(
    tokenize_pairs, batched=False, remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)

tokenized_val = val_dataset.map(
    tokenize_pairs, batched=False, remove_columns=val_dataset.column_names,
    desc="Tokenizing val"
)

tokenized_train.save_to_disk("tokenized_combined_dialog_data/train")
tokenized_val.save_to_disk("tokenized_combined_dialog_data/val")