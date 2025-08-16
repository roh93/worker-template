import os
import runpod

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "defog/sqlcoder-7b-2")
DTYPE = os.getenv("TORCH_DTYPE", "float16")  # "float16" or "bfloat16"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is not None:
        return

    torch_dtype = torch.float16 if DTYPE == "float16" else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)

def build_prompt(question: str, schema: str) -> str:
    # Per model card guidance: structured prompt + beam search, no sampling.
    # https://huggingface.co/defog/sqlcoder-7b-2 (Prompt section)
    return (
        "### Task\n"
        f"Generate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n"
        "### Database Schema\n"
        "The query will run on a database with the following schema:\n"
        f"{schema}\n\n"
        "### Answer\n"
        f"Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]\n"
        "[SQL]"
    )

def generate_sql(question: str, schema: str, max_new_tokens: int = 256):
    load_model()
    prompt = build_prompt(question, schema)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,     # recommended
        num_beams=4,         # recommended
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    out_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Return only the SQL after the "[SQL]" tag if present
    sql = out_text.split("[SQL]", 1)[-1].strip()
    return {"sql": sql, "prompt": prompt}

def handler(event):
    """
    Expects:
    {
      "input": {
        "question": "...natural language question...",
        "schema": "...DDL or tables/columns text...",
        "max_new_tokens": 256   # optional
      }
    }
    """
    try:
        inp = event.get("input", {}) or {}
        question = inp.get("question")
        schema = inp.get("schema")
        max_new = int(inp.get("max_new_tokens", 256))

        if not question or not schema:
            return {"error": "Both 'question' and 'schema' are required."}

        result = generate_sql(question, schema, max_new_tokens=max_new)
        return {"status": "COMPLETED", "output": result}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

runpod.serverless.start({"handler": handler})
