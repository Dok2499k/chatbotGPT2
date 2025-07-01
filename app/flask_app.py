from flask import Flask, render_template, request, jsonify, session
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch, uuid
from pathlib import Path

app = Flask(__name__)
app.secret_key = "supersecret"
MAX_RETRIES = 2

model_path = Path(__file__).parent.parent/"final_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(Path(__file__).parent.parent/"final_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

dialogue_cache = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("user_input", "").strip()
    if not user_input:
        return jsonify({"error": "Пустой запрос"}), 400

    user_id = session.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
        session["user_id"] = user_id

    history = dialogue_cache.get(user_id, [])
    history.append(user_input)

    retries = 0
    generated_text = ""

    while retries <= MAX_RETRIES:
        recent_history = history[-2:]
        prompt = "\n".join(recent_history) + "\n"

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_ids = input_ids[:, -700:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        if len(generated_text) > 2:
            break

        retries += 1

    if not generated_text:
        generated_text = "I'm sorry, I didn't catch that. Could you please rephrase?"

    history.append(generated_text)
    dialogue_cache[user_id] = history

    return jsonify({"response": generated_text})

@app.route("/reset", methods=["POST"])
def reset():
    user_id = session.get("user_id")
    if user_id and user_id in dialogue_cache:
        dialogue_cache.pop(user_id)
    return jsonify({"status": "History reset successfully"})

if __name__ == "__main__":
    app.run(debug=True)