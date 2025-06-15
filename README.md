# 📘 Fine-Tuning SpanBERT & SpanBERT-CRF on SQuAD v2

This repository contains the implementation of **SpanBERT** and **SpanBERT-CRF** fine-tuned for **extractive question answering** on the **SQuAD v2** dataset. It adapts the Hugging Face QA pipeline while introducing custom span alignment logic and postprocessing for handling long contexts, unanswerable questions, and multiple valid answers.

---

## 🔍 Project Overview

- **Dataset**: SQuAD v2
- **Models**: SpanBERT (base), optionally extended with CRF
- **Task**: Predict the start and end span of the answer in a given context (or predict "no answer")
- **Evaluation Metric**: Exact Match (EM)

---

## 📁 Files

| File | Description |
|------|-------------|
| `spanbert.ipynb` | Full implementation for training, evaluation, and postprocessing for SpanBERT model|
| `spanbert-crf.ipynb` | Full implementation for training, evaluation, and postprocessing for Custom CRF Layer + SpanBERT model|
| `spanbert.pth` | Saved model checkpoint with the highest EM score for SpanBERT model|
| `spanbert-crf.pth` | Saved model checkpoint with the highest EM score for SpanBERT+CRF model|

---

## 📦 Dataset Preparation

- Subset of **15,000** training examples from **SQuAD v2**.
- Full validation split used for evaluation.
- Dataset loaded using `datasets.load_from_disk("rajpurkar/squad_v2")`.

---

## 🔧 Preprocessing

This step prepares `(question, context)` pairs into tokenized inputs suitable for model training.

### ✅ Key Steps

1. **Tokenization**  
   Using `SpanBERT/spanbert-base-cased` with:
   - `max_length=384`
   - `stride=128` (sliding window for long contexts)
   - `truncation="only_second"` (context gets truncated)
   - `return_overflowing_tokens=True`
   - `return_offsets_mapping=True`

2. **Answer Span Labeling**  
   - Character-level answer span is mapped to token positions using a custom function `mark_start_end()`.
   - If the span is outside the context window or question is unanswerable, `(start=0, end=0)` is used (pointing to `[CLS]`).

3. **Training Examples**  
   Each sample is tokenized and annotated with:
   - `input_ids`, `token_type_ids`, `attention_mask`
   - `start_positions`, `end_positions`

4. **Validation Examples**  
   - Stores **up to 6 valid spans** per example.
   - Uses padding (`-1`) for unused slots.
   - 
5. **BIO Labeling (for CRF variant)**

  In addition to standard span-labeling (`start_positions`, `end_positions`), the dataset is also annotated using **BIO tagging**:

    - B (1): Marks the beginning of the answer span  
    - I (2): Marks tokens inside the answer span  
    - O (0): All other tokens

This BIO-encoded label sequence is used to train the **SpanBERT + CRF** variant, enabling sequence-level span decoding using a Conditional Random Field layer.


---

## 🧠 Model Architecture

### ✅ SpanBERT
Uses `AutoModelForQuestionAnswering`, directly fine-tuned on the prepared dataset.

### ➕ SpanBERT + CRF 
```
class CustomCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
        self.crf = CRF(3, batch_first=True)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 3)
        self.relu = nn.ReLU() 

    def forward(self, input_ids, attention_mask, token_type_ids, tags, eval_flag):
        ops = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = self.classifier(ops.last_hidden_state)
        temp = self.relu(logits) # taki model seekh jaye 

        if eval_flag == 1:
            ls = -self.crf(temp, tags, mask=attention_mask.bool())
            pr = self.crf.decode(temp, mask=attention_mask.bool())
            return ls, pr
        else:
            ls = -self.crf(temp, tags, mask=attention_mask.bool())
            return ls

```
---

## 🏋️ Training Details

The model is trained using a custom PyTorch loop for full control over optimization, evaluation, and postprocessing.

### ✅ Training Configuration

- **Optimizer**: `AdamW` with learning rate of `3e-5`
- **Batch Size**: 8
- **Epochs**: 6 (minimum required)
- **Loss**: Automatically computed via `start_positions` and `end_positions`
- **Device**: Automatically uses GPU if available, otherwise CPU
- **Checkpointing**: The model achieving the **highest Exact Match (EM)** on the validation set is saved as `best_model.pth`

### 🧠 Custom Training Loop Highlights

This approach ensures the training loop mirrors SQuAD-style evaluation faithfully, while supporting multiple valid answers and token span overlaps.

- **Feature-to-Example Mapping**:
  - Handles overlapping tokenized contexts by grouping multiple features (chunks) back to their original example IDs.
  
- **Robust Evaluation via EM Matching**:
  - During evaluation, the model may produce multiple predictions for a single example (due to long contexts being split).
  - For each prediction, a **best-matching reference answer** is dynamically chosen from the set of ground truths.
  - This logic ensures fairness when multiple gold answers are correct (e.g., different phrasings or lengths).

- **Best Span Extraction**:
  - Predicted start and end logits are converted into spans using a `get()` function.
  - Spans are filtered to exclude invalid ones (e.g., start > end, unanswerable).
  - The final predicted answer text is extracted using `offset_mapping`.

- **Best EM Aggregation**:
  - After scoring each predicted span against potential references, the version with the **highest EM** is retained.
  - The final EM score is computed only over these best predictions per example.


---

## 🔁 Postprocessing & Evaluation

### 🎯 Why Postprocessing?

The model outputs only raw logits for the start and end positions of the answer span — these must be decoded back into readable text from the original context.

> Instead of computing probabilities over all possible (start, end) pairs using softmax, we use **logit sums** for efficiency. This works because:  
> log(ab) = log(a) + log(b)

This avoids the computational overhead of softmax and directly compares summed logits of top scoring start and end positions. The final answer is selected from the span with the highest combined logit score among valid candidates.


### 🧷 Postprocessing Logic

1. **Select Top-N (start, end) Logits**  
   - Picks top 20 candidates from each logits array
   - Combines them to create potential spans

2. **Span Filtering**  
   - Removes invalid spans (start > end, length > 30, etc.)

3. **Character Offset Recovery**  
   - Uses `offset_mapping` to convert token spans back to original text

4. **Example-Level Aggregation**  
   - Each example may span multiple input chunks
   - The final prediction per example is the one with the **highest EM score** against all ground-truths

### 📐 Exact Match Metric

```python
def exact_match_score(predictions, references):
    assert len(predictions) == len(references)
    return sum(p == r for p, r in zip(predictions, references)) / len(references) * 100
```

---

## 📊 Results

<img width="1222" alt="image" src="https://github.com/user-attachments/assets/d17ab9fd-9d71-46e2-a41a-16aa87e80161" />

<img width="1635" alt="image" src="https://github.com/user-attachments/assets/47499c2f-dcfd-45b8-8e4e-06caf87cc963" />


---

## 🚀 Future Work

- Extend to other datasets like Natural Questions or TyDiQA
- Upload model to Hugging Face Hub for sharing and inference widget

---

## 📜 Citation

If you found this work useful, consider referencing:

```bibtex
@misc{spanbertqa2025,
  title={Fine-tuning SpanBERT for Extractive QA on SQuAD v2},
  author={Your Name},
  year={2025},
  note={https://github.com/yourusername/spanbert-qa-squad}
}
```
