# Why use Transformers instead of an inference server like vLLM

- `vLLM` is an inference server designed to serve models over HTTP long-term.
- `transformers` is a library that gives you direct programmatic access to load and run models in Python.

## Why transformers are better for this lab

This lab is a quantization comparison experiment, not a production inference deployment. You need to:

- Load the model in a specific precision (`FP16`, `INT8`, `INT4`)
- Measure exact VRAM usage at each stage
- Control exactly how the model loads
- Intentionally stress it to demonstrate OOM behavior

`transformers` gives you that fine-grained control.

### Example loading modes
```python
# FP16
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)

# INT8
model = AutoModelForCausalLM.from_pretrained(model, load_in_8bit=True)

# INT4
model = AutoModelForCausalLM.from_pretrained(model, load_in_4bit=True)
```

`vLLM` abstracts much of this away. That means it is harder to answer questions like:

- "Load this model in INT8 and tell me exactly how much VRAM it used."

## When to use each

| Scenario | Best tool |
|---|---|
| Research, benchmarking, quantization experiments | `transformers` |
| Production inference, HTTP API, batching, high throughput | `vLLM` |
| Fine-tuning / training | `transformers` + `PEFT` |

Think of it this way:

- `transformers` is the scalpel for precise control.
- `vLLM` is the production engine for serving at scale.
