# GPT-2 Conversational Chatbot

## Introduction 

The project outlined in this report focuses on creating a conversational AI agent—an intelligent chatbot—using GPT-2 fine-tuned on curated datasets of human dialogue. The end goal is to deploy a multi-turn, context-aware chatbot accessible through a web interface.

This report documents the complete lifecycle of the chatbot, from problem formulation and data preparation to model training, evaluation, and deployment via a Flask web application. It is intended as a comprehensive technical report, complete with motivation, design decisions, implementation details, results, and conclusions.

---

## Project Objectives

The key goals of the project are:

- Fine-tune OpenAI's GPT-2 model to generate coherent, persona-consistent responses in multi-turn conversations.
- Build a user-friendly, responsive web interface using Flask.
- Maintain session-level context for each user using server-side memory.
- Implement robust prompt formatting and decoding strategies to enhance dialogue quality.
- Integrate real-time inference with caching mechanisms to ensure responsiveness.

---

## Background

Chatbots have become indispensable in domains ranging from customer service to personal assistance. While rule-based bots are rigid and limited, generative models like GPT-2 provide the flexibility to respond creatively and contextually.

GPT-2, developed by OpenAI, is a transformer-based model pre-trained on a massive text corpus. While GPT-2 is powerful, it is not tailored out-of-the-box for conversation. Fine-tuning it on dialogue datasets allows it to learn patterns of human interactions, making it suitable for chatbot development.

The challenge lies in customizing GPT-2 for dialogue generation, optimizing the model for coherence, persona-consistency, and responsiveness, and finally integrating it into a web-based interface.

---

## Datasets Used

To equip GPT-2 with conversational capabilities, we used two widely accepted dialogue datasets:

### 1. Cornell Movie Dialogues Corpus

- Source: Hugging Face Datasets
- Nature: Scripted dialogues from movie scripts
- Format: Multi-turn exchanges between characters
- Use: Extracted input-response pairs for general conversational structure

### 2. PersonaChat Dataset

- Source: Facebook AI via Hugging Face
- Nature: Persona-grounded dialogue
- Format: Multi-turn chat sessions with personal context
- Use: Added depth and persona-awareness to the chatbot’s responses

Both datasets were used to prepare high-quality input-target pairs suitable for fine-tuning GPT-2.

---

## Methodology

### 1. Data Preparation

Implemented in `prepare_data.py`, this stage includes:

- **Loading datasets**: Used `convokit` and `datasets` to load Cornell and PersonaChat.
- **Filtering and formatting**: Extracted pairs of utterances where both input and output are non-empty.
- **Combining**: Merged both datasets into a unified format using Hugging Face `Dataset` API.
- **Tokenization**: Used GPT-2 tokenizer with EOS token and max length padding.
- **Saving**: Stored processed datasets to disk using `save_to_disk()` for reuse.

This step produces:

```
tokenized_combined_dialog_data/
├── train/
└── val/
```

---

### 2. Model Fine-Tuning

We've implemented two scripts: `train_model.py` and `resume_train.py`.

- **Initial Training**: Used `train_model.py` to train GPT-2 on the prepared dataset for 3 epochs.
- **Checkpointing**: Model checkpoints were saved every 10,000 steps.
- **Resuming Training**: `resume_train.py` was used to resume from a checkpoint if training was interrupted.
- **Hyperparameters**:
  - Batch size: 8
  - Learning rate: 5e-5
  - Max length: 256 tokens
  - Evaluation strategy: Steps (every 5000)
  - Logging: TensorBoard support enabled
- **Mixed Precision**: FP16 training enabled for GPU acceleration.

Output:

```
final_model/
├── config.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
```

---

### 3. Session Management and Inference

`run_inference.py` serves as a local CLI interface:

- Loads the trained model and tokenizer from `final_model`
- Maintains multi-turn conversation with input prompts
- Applies top-p (nucleus sampling), top-k, and temperature sampling
- Continuously accepts user input until 'exit' is typed

Prompt format:

```
You: Hello
🤖 GPT-2: Hi! How can I help you today?
```

---

### 4. Web Application Integration

Implemented in `flask_app.py`:

- Serves a Flask app with HTML frontend (`index.html`)
- Routes:
  - `/` — loads the UI
  - `/predict` — handles AJAX requests for chatbot replies
  - `/reset` — clears session history
- Uses Flask `session` to maintain user-specific chat history
- Formats input using last 2 utterances to build context-aware prompts
- Applies decoding strategies identical to CLI

HTML UI includes input field, send/reset buttons, and chat history rendering. The app is designed to be minimal but functional.

---

## Tools and Technologies

| Category          | Tools Used                         |
| ----------------- | ---------------------------------- |
| Programming       | Python 3.12                        |
| ML Frameworks     | PyTorch, Hugging Face Transformers |
| Dataset Libraries | `datasets`, `convokit`             |
| Tokenization      | GPT2Tokenizer                      |
| Web Framework     | Flask                              |
| UI Templates      | HTML, JavaScript                   |
| Logging           | TensorBoard                        |
| Dev Environment   | PyCharm IDE                        |

---

## Results and Observations

The chatbot successfully generates context-aware, coherent responses when fine-tuned on Cornell and PersonaChat data. Key results include:

- **Dialogue Quality**: Improved significantly after training vs base GPT-2.
- **Persona Retention**: Notable consistency when trained with PersonaChat.
- **Responsiveness**: Inference latency is minimal with GPU; under 2 seconds per response.
- **Sampling Effects**: Top-p and temperature tuning reduced repetition.
- **Multi-turn Consistency**: Session-level memory retained up to last 2 turns, producing believable continuity.

Example interaction:

```
User: Hi, how are you?
GPT-2: I'm well, and you?
User: What sports do you prefer?
GPT-2: I love basketball! I love riding my bike.
```

---

## Challenges and Solutions

| Challenge                              | Solution                                                          |
| -------------------------------------- | ----------------------------------------------------------------- |
| Slow dataset processing                | Used `tqdm` and batching for efficient loading                    |
| Large memory footprint during training | Applied FP16 and batch size tuning                                |
| Empty or short generations             | Implemented retry mechanism and fallbacks                         |
| Session loss in Flask                  | Used `Flask.session` and in-memory cache (`dialogue_cache`)       |
| Prompt length overflow                 | Truncated to last 700 tokens to fit within 1024-token GPT-2 limit |

---

## Conclusion

This project demonstrates a successful pipeline for building a conversational agent using GPT-2. From data preparation and training to deployment via a web interface, the system meets all functional objectives outlined in the proposal:

- Multi-turn, session-aware conversations
- Real-time Flask-based UI
- GPT-2 fine-tuned for dialogue tasks

The chatbot exhibits robust performance, delivering high-quality and persona-consistent replies. The modular architecture makes it easily extendable and adaptable for future enhancements.

---

## Possible Improvements

While the system is functional, several enhancements can be considered:

- **Web UI improvements**: Add styling, typing animation, and scrollback.
- **Model Export**: Convert to ONNX or TorchScript for deployment efficiency.
- **Persona injection**: Dynamically customize persona based on user input.
- **Logging & analytics**: Track queries and responses for evaluation.
- **Deployment**: Dockerize and deploy on cloud platforms like Render or HF Spaces.
- **Chat Summarization**: Add summarization at session end for context handover.

---

**Authors:** Doston Kanaev and Saidamir Rustamov\
**Date:** July 2025\
**Model Version:** GPT-2 fine-tuned on Cornell + PersonaChat

