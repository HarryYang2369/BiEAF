# BiEAF: A Bidirectional Enhanced Attention Flow Model for Question Answering

This repository contains the reference implementation of **BiEAF (Bidirectional Enhanced
Attention Flow)**, a span-extraction question-answering model for the
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

BiEAF is built on top of [BiDAF (Seo et al., 2016)](https://arxiv.org/abs/1611.01603) and
[the Transformer's self-attention (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).
Where BiDAF only models the **inter-sentence** correlation between context and query, BiEAF
adds an **intra-sentence self-attention** step so the model can also weight the most
informative tokens *within* each sentence before the cross-sentence attention is computed.
This is the "enhanced attention flow" layer.

> Yihan Yang. *BiEAF: An Bidirectional Enhanced Attention Flow Model for Question Answering
> Task.* 2021 2nd International Conference on Information Science and Education (ICISE-IE),
> pp. 344–348. DOI: 10.1109/ICISE-IE53922.2021.00086. (`BiEAF.pdf` is included in this repo.)

---

## Table of contents

- [Task](#task)
- [Model architecture](#model-architecture)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Paper ↔ code correspondence](#paper--code-correspondence)
- [Known errata in the paper](#known-errata-in-the-paper)
- [Citation](#citation)
- [License](#license)

---

## Task

Given a **context** paragraph and a **question**, the model predicts the answer as a
contiguous **span** of the context — i.e. a `start` token index and an `end` token index.
This is the extractive QA setting popularized by SQuAD.

---

## Model architecture

BiEAF follows an **encoder → enhanced-attention-flow → decoder** design. The full layout is
shown in Figure 2 of the paper; the implementation lives in [`model/model.py`](model/model.py).

```
        Context words                         Query words
             │                                    │
   ┌─────────▼─────────┐                ┌─────────▼─────────┐
   │  Char-CNN + GloVe │                │  Char-CNN + GloVe │   (1) Word/Char Embedding
   │   + Highway net   │                │   + Highway net   │
   └─────────┬─────────┘                └─────────┬─────────┘
             │                                    │
        BiLSTM (shared)  ──────────────────  BiLSTM (shared)      (2) Contextual Embedding
             │  h_c                               │  h_q
   ┌─────────▼─────────┐                ┌─────────▼─────────┐
   │  Self-Attention   │                │  Self-Attention   │   (3a) Intra-sentence
   │   g_c = s_c ⊙ h_c │                │  g_q = s_q ⊙ h_q  │        self-attention
   └─────────┬─────────┘                └─────────┬─────────┘
             └────────────────┬───────────────────┘
                              ▼
                  Cross-Attention (Enhanced Attention Flow)        (3b) Inter-sentence
                       f_c = Σ α'_c · g_q                              attention
                              │
                     BiLSTM₁ → BiLSTM₂  ──► G                     (4) Modeling Layer
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        Dense → P_start             BiLSTM → Dense → P_end        (5) Output / Decoder
```

### 1. Encoder (`§III-A`)

- **Word embeddings:** pretrained, frozen [GloVe](https://nlp.stanford.edu/projects/glove/)
  (`6B`, `word_dim` dimensions).
- **Character embeddings:** a 1-D char CNN (`char_dim` → `char_channel_size` via a
  `char_channel_width` kernel + max-pool), so the model can handle rare/OOV tokens.
- **Highway network:** fuses the word and char representations (2 layers).
- **Contextual embedding:** a shared bidirectional LSTM produces the contextual
  representations `h_c` (context) and `h_q` (query):

  ```
  h_c, h_q = BiLSTM(x_c), BiLSTM(x_q)
  ```

### 2. Enhanced Attention Flow (`§III-B`)

This is the core contribution. It runs in two stages.

**(a) Intra-sentence self-attention** — re-weights each position by a learned scalar
importance score so informative tokens are emphasized and noise is suppressed:

```
α   = σ(W · h + b)        # scalar score per position (σ = ReLU)
s   = softmax(α)          # normalized weights over the sequence
g   = s ⊙ h               # importance-weighted representation
```

applied independently to the context (`h_c → g_c`) and the query (`h_q → g_q`).

**(b) Inter-sentence cross-attention** — links and fuses context and query information,
producing the query-aware context representation `f_c`:

```
α'_c = σ(g_c · g_qᵀ)      # dot-product similarity (σ = ReLU)
α'_c = softmax(α'_c)      # normalize over query positions
f_c  = Σ α'_c · g_q       # attended query representation
```

> **Note:** the published paper contains a typo here — it writes `f_c = Σ α'_c · α'_c`. The
> correct formula (and what the code implements) is **`f_c = Σ α'_c · g_q`**: the second
> factor is the *query representation being attended over*, not `α'_c`. See
> [Known errata](#known-errata-in-the-paper).

### 3. Decoder / Output (`§III-C`)

- **Modeling layer:** two stacked bidirectional LSTMs over `f_c` produce `G`, capturing the
  interaction among context words conditioned on the query:

  ```
  G = BiLSTM₁(BiLSTM₂(f_c))
  ```

- **Output layer:** two classifiers predict the start and end indices. The start logits are a
  dense projection of `G`; the end logits are a dense projection of `G` after an additional
  BiLSTM (as in Figure 2 of the paper).

  ```
  P_start = softmax(W_s · G)
  P_end   = softmax(W_e · output_LSTM(G))
  ```

  The model returns **raw logits**; the `softmax`/`log-softmax` is applied by
  `CrossEntropyLoss` during training and by `LogSoftmax` during inference (`run.py`).

---

## Repository layout

```
.
├── BiEAF.pdf            # the paper
├── run.py               # training + evaluation entry point
├── evaluate.py          # SQuAD v1.1 official EM / F1 scoring
├── model/
│   ├── model.py         # the BiEAF network (encoder, EAF layer, decoder)
│   ├── data.py          # SQuAD loading, tokenization, vocab, iterators
│   └── ema.py           # exponential moving average of parameters
├── utils/
│   └── nn.py            # LSTM and Linear wrappers (packed-sequence handling, init)
├── prediction0.out      # example prediction file (id → answer span)
├── requirements.txt
└── LICENSE              # MIT
```

---

## Installation

The code targets **Python 3.6** and the (older) library versions pinned in
[`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

```
torch==0.4.0
nltk==3.4.5
tensorboardX==0.8
torchtext==0.2.3
```

You also need the NLTK tokenizer data:

```python
import nltk
nltk.download('punkt')
```

> These pins are old. If you run on a modern stack, expect to update the `torchtext` data
> API usage in `model/data.py` (the `Field`/`BucketIterator` API changed substantially after
> `torchtext` 0.4).

---

## Data preparation

Download the SQuAD v1.1 files and place them under `.data/squad/`:

```
.data/squad/train-v1.1.json
.data/squad/dev-v1.1.json
```

On first run, `model/data.py` preprocesses these into JSON-lines (`*.jsonl`) and caches the
torchtext examples under `.data/squad/torchtext/`. GloVe vectors are downloaded automatically
by torchtext on first use.

> **Important:** `model/data.py` currently caps the data to the first 100 examples for fast
> debugging:
> ```python
> train_examples = train_examples[:100]
> dev_examples   = dev_examples[:100]
> ```
> Remove these two lines to train/evaluate on the full dataset and reproduce the paper's
> numbers.

---

## Training

```bash
python run.py
```

Key command-line arguments (see `main()` in [`run.py`](run.py) for the full list):

| Argument | Default | Description |
|---|---|---|
| `--epoch` | `12` | number of training epochs |
| `--train-batch-size` | `60` | training batch size |
| `--dev-batch-size` | `100` | evaluation batch size |
| `--learning-rate` | `0.5` | Adadelta learning rate |
| `--hidden-size` | `100` | LSTM hidden size (per direction) |
| `--word-dim` | `100` | GloVe embedding dimension |
| `--char-dim` | `8` | character embedding dimension |
| `--char-channel-size` | `100` | char-CNN output channels |
| `--char-channel-width` | `5` | char-CNN kernel width |
| `--dropout` | `0.2` | dropout probability |
| `--context-threshold` | `400` | drop training contexts longer than this |
| `--exp-decay-rate` | `0.999` | EMA decay for parameter averaging |
| `--gpu` | `0` | GPU id (falls back to CPU automatically) |

Training optimizes the sum of two cross-entropy losses (start + end). An
**exponential moving average** (`model/ema.py`) of the parameters is maintained and used at
evaluation time. The best model (by dev F1) is saved to `saved_models/BiEAF_<time>.pt`, and
TensorBoard logs are written to `runs/`.

---

## Evaluation

Evaluation runs automatically during training (every `--print-freq` steps) using the official
SQuAD v1.1 **Exact Match (EM)** and **F1** metrics implemented in
[`evaluate.py`](evaluate.py).

At inference, span selection (`run.py`) scores every `(start, end)` pair via
`log P_start + log P_end`, masks invalid spans where `end < start`, and picks the
highest-scoring valid span. Predictions are written to `prediction<gpu>.out` as a
`{question_id: answer_text}` JSON map (see `prediction0.out` for an example).

---

## Results

Reported on the SQuAD v1.1 dev set (paper Table I):

| Model | Exact Match | F1 |
|---|---|---|
| Logistic Regression | 40.4 | 51.0 |
| Dynamic Chunk Reader | 62.5 | 71.0 |
| Fine-Grained Gating | 62.5 | 73.3 |
| Match-LSTM | 64.7 | 73.7 |
| Multi-perspective Matching | 65.5 | 75.1 |
| Dynamic Coattention Networks | 66.2 | 75.9 |
| R-Net | 68.4 | 77.5 |
| BiDAF | 68.0 | 77.3 |
| **BiEAF (ours)** | **68.5** | **77.7** |

**Ablation study** (paper Table II; `(-)` = component removed):

| Variant | Exact Match | F1 |
|---|---|---|
| Full model | 68.5 | 77.7 |
| (-) Char embedding | 67.1 | 75.0 |
| (-) Modeling layer | 62.7 | 73.3 |
| (-) Enhanced attention flow | 60.3 | 69.1 |

Removing the enhanced attention flow causes the largest drop (~8 EM / ~9 F1), confirming that
combining intra- and inter-sentence attention is the most important component.

---

## Paper ↔ code correspondence

| Paper (§) | Equation / component | Code |
|---|---|---|
| III-A | `x = GloVe(C/Q)` | `model.py:21, 126–127` |
| III-A | Char-CNN + highway (Fig. 2) | `model.py:13–18, 66–84` |
| III-A | `h_c, h_q = BiLSTM(x_c), BiLSTM(x_q)` | `model.py:136–137` |
| III-B | self-attention `g = softmax(σ(W·h+b)) ⊙ h` | `self_att_layer`, `model.py:86–96` |
| III-B | cross-attention `f_c = Σ α'_c · g_q` | `eaf_layer`, `model.py:98–113` |
| III-C | `G = BiLSTM₁(BiLSTM₂(f_c))` | `model.py:147` |
| III-C | `P_start`, `P_end` | `output_layer`, `model.py:115–120` |

The implementation faithfully reproduces the paper's method. The few places where the code is
more detailed than the §III equations (char-CNN, highway network, the extra BiLSTM on the end
path) are all consistent with **Figure 2** and the ablation study; the equations are simply
abbreviated. The `softmax`/`σ` shown in the output equations are applied by the loss /
inference code rather than inside the network.

---

## Known errata in the paper

1. **`f_c` formula (end of §III-B).** The paper prints `f_c = Σ α'_c · α'_c`. This is a typo.
   The correct expression — and what the code implements — is:

   ```
   f_c = Σ α'_c · g_q
   ```

   i.e. the cross-attention output is a weighted sum over the **query representations** `g_q`,
   not over the attention weights themselves.

2. **Self-attention formula (§III-B).** The paper prints `g = Σ α · s`, which (taken
   literally) would collapse the sequence to a scalar. The intended/implemented operation is
   an element-wise re-weighting that preserves the sequence: `g = softmax(σ(W·h+b)) ⊙ h`.

---

## License

Released under the [MIT License](LICENSE).
