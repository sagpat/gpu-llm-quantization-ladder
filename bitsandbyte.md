# BitsAndBytes: Quantization in Action

## Phase 1 vs Phase 2: What Changed

### FP16 (Phase 1)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # ← explicit FP16
)
```
Every weight stored as a 16-bit float. 7 billion parameters × 2 bytes = ~14 GB.

### INT8 (Phase 2)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,  # ← this one flag does everything
    device_map="cuda",
)
```
Every weight stored as an 8-bit integer. 7 billion parameters × 1 byte = ~7 GB.

That one flag `load_in_8bit=True` is the entire difference. Everything else — tokenizer, inference loop, accuracy test — is identical.

## What BitsAndBytes Actually Does

Without bitsandbytes, `load_in_8bit=True` would just crash — HuggingFace transformers doesn't know how to do INT8 quantization by itself. bitsandbytes is the engine that makes it work.

### The Problem: Naive Approach (Wrong)
- FP16 weight: 0.3842
- → just cast to INT8: 38 (multiply by 100, round)
- → cast back for compute: 0.38
- → error: 0.0042 ← small but multiplied across 7B params = garbage output

If you just truncate every weight to 8-bit, the accumulated rounding errors across 7 billion parameters destroy the model's ability to reason. You'd get nonsense outputs.

### The Solution: LLM.int8()
It uses a technique called **mixed-precision decomposition**, discovered by Tim Dettmers in 2022:

1. **Find the outliers**
   - During loading, bitsandbytes scans each layer and identifies "outlier" weights — values that are much larger than the rest (typically the top 0.1% of values).
   - These outliers carry disproportionate importance to the model's accuracy.

   ```
   Layer weights: [0.12, 0.08, 0.31, 0.09, 8.74, 0.15, 0.22, 0.07, 7.91, ...]
                                             ^^^^                   ^^^^
                                           outliers — kept in FP16
   ```

2. **Split into two streams**
   - Outlier weights (0.1%) → kept in FP16 → multiplied in FP16
   - Normal weights (99.9%) → quantized to INT8 → multiplied in INT8

3. **Recombine**
   - FP16 result + INT8 result = final output

The final result is nearly identical to running everything in FP16, because the outliers (which matter most) were never quantized.

**Visually:**
```
7B parameters
│
├── 6.993B normal weights → INT8 → 6.993 GB
│                                  (was 13.986 GB in FP16)
│
└── 0.007B outlier weights → FP16 → 0.014 GB
                                    (stays precise)
│
Total: ~7 GB instead of ~14 GB
Accuracy loss: <1% on most benchmarks
```

## The Accuracy Test in Phase 2

This is why the 5 questions are there:
```python
questions = [
    ("What is the capital of France?",        "Paris"),
    ("What is 15 multiplied by 8?",           "120"),
    ("Who wrote Romeo and Juliet?",           "Shakespeare"),
    ("What planet is closest to the sun?",   "Mercury"),
    ("What is the chemical symbol for gold?", "Au"),
]
```
These test whether quantization damaged the model's knowledge. If bitsandbytes works correctly, you should get 5/5. If naive quantization was used, you'd get 2/5 or worse.

## The Full Comparison You're Building

| Precision | Storage per weight | Expected VRAM | Technique | Library       | Accuracy loss |
|-----------|-------------------|---------------|-----------|---------------|---------------|
| FP16      | 2 bytes          | ~14 GB       | None     | transformers | baseline     |
| INT8      | 1 byte           | ~7 GB        | LLM.int8()| bitsandbytes | <1%         |
| INT4      | 0.5 bytes        | ~4 GB        | NF4      | bitsandbytes | 1-3%        |

Each phase halves the memory. Phase 3 will use `load_in_4bit=True` with a slightly different quantization technique called NF4 (Normal Float 4) — also from bitsandbytes.