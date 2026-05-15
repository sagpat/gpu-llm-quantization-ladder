import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("Attempting to load Mistral 7B in FP16...")

# Initialize variables
model = None
tokenizer = None

try:
    # 1. Load the Model
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="cuda",
        token=os.environ.get("HF_TOKEN")
    )
    print(f"Loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB used")
    
    # 2. Load the Tokenizer
    print("\n=== Loading Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        token=os.environ.get("HF_TOKEN")
    )
    
except Exception as e:
    used = torch.cuda.memory_allocated()/1e9
    print(f"FAILED: {e}")
    print(f"Memory at crash: {used:.2f} GB / 15.8 GB")
    print("Try doing model quantization.")

# 3. Run Test Generation Loop
if model is not None and tokenizer is not None:
    print("\n=== Testing Inference Generation ===")
    
    # Define a prompt using Mistral's required chat template format
    prompt = "Explain quantum computing in one simple sentence."
    messages = [{"role": "user", "content": prompt}]
    
    # Format prompt for the instruct model
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    
    print(f"Prompt: {prompt}")
    print("Generating response...")
    
    try:
        # Generate text tokens
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=50, 
                do_sample=True,
                temperature=0.7
            )
        
        # Decode and print output (skipping the prompt tokens)
        generated_tokens = outputs[0][inputs.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"\nResponse:\n{response}")
        print(f"\nFinal VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
    except Exception as gen_error:
        print(f"Generation failed due to: {gen_error}")
        print("This is likely an Out-Of-Memory (OOM) error during the forward pass.")
