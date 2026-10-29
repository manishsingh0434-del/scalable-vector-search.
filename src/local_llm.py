
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"


def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_answer(question, passages, tokenizer, model):
    if not passages:
        return "I don't know based on the provided documents."

    context = "\n".join(passage[:600] for passage in passages[:3])

    prompt = (
        "Answer using only the context. If the answer is missing, "
        "say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=2,
        )

    return tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).strip()
