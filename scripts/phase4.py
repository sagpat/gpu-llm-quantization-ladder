import torch, time, os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
token = os.environ.get("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

print("Loading Mistral 7B in INT4 (NF4)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda",
    token=token
)

mem = torch.cuda.memory_allocated()/1e9
print(f"Memory used: {mem:.2f} GB")
print(f"Memory saved vs FP16 (~15GB): {15.0 - mem:.1f} GB saved")
print(f"Memory saved vs INT8 (~8GB): {8.0 - mem:.1f} GB saved")

questions = [
    ("What is the capital of France?",        "Paris"),
    ("What is 15 multiplied by 8?",           "120"),
    ("Who wrote Romeo and Juliet?",           "Shakespeare"),
    ("What planet is closest to the sun?",   "Mercury"),
    ("What is the chemical symbol for gold?", "Au"),
]

correct = 0
latencies = []
for q, expected in questions:
    inputs = tokenizer(f"Answer in one word or number: {q}",
                       return_tensors="pt").to("cuda")
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    lat = (time.time()-start)*1000
    latencies.append(lat)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    got = expected.lower() in ans.lower()
    if got: correct += 1
    print(f"Q: {q} | Correct: {got} | {lat:.0f}ms")

print(f"INT4 Accuracy: {correct}/5 = {correct/5*100:.0f}%")
print(f"Avg latency: {sum(latencies)/len(latencies):.0f}ms")
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")