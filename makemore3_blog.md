# Blog: Calibrating Makemore 3

## 1. Why revisit Makemore?
Makemore 3 picks up after the bigram/trigram language model to prepare for the RNN / GRU / LSTM experiments hinted at in the intro notes. Recurrent nets are universal approximators, but they become brittle under naive initialization. Before graduating to sequences, this notebook uses a trigram character model to understand how activations, gradients, and normalization behave. Everything runs in PyTorch with `%matplotlib inline` and a default `plt.rcParams['figure.figsize'] = [6, 4]`.

## 2. Names dataset and trigram context
We read `names.txt`, build a character vocabulary (with `'.'` mapped to index `0`), call `len(words)` to sanity-check the corpus size, and prepare train/dev/test splits with an 80/10/10 shuffle. A fixed context of three characters (`block_size = 3`) feeds the model.

```python
with open("names.txt", "r") as file:
    words = file.read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)
```

Splits are produced with `random.seed(42)` and `random.shuffle(words)`, giving tensors such as `Xtr.shape == (num_examples, 3)` and embeddings via `C[Xb]`.

## 3. Baseline trigram MLP
The baseline is an embedding layer followed by a single hidden `tanh` layer (`n_embd=10`, `n_hidden=200`) and an output linear layer over the vocabulary. Parameters are tracked in a list for manual SGD updates and `sum(p.nelement() for p in parameters)` confirms the count. Training runs for up to `max_steps = 200000` updates, samples 32 examples per step, and stores log-losses in `lossi` for plotting (`plt.plot(lossi)`).

```python
n_embd, n_hidden = 10, 200
g = torch.Generator().manual_seed(2147483647)

C  = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    emb = C[Xb]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 1_000_000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
```

`split_loss` runs under `@torch.no_grad()` for train/dev/test splits (loss is the *average* over the minibatch), and the sampling loop (`torch.multinomial`) produces 20 names by iteratively feeding back predictions.

## 4. Expected vs. observed loss
With 27 possible characters, the first minibatch should have cross-entropy `-log(1/27) ≈ 3.30`. Remember that the reported value is the average of `-log p[target]` per example, not the sum across the vocabulary. The notebook verifies the expectation explicitly and shows how random logits make the model “confidently wrong.”

```python
expected = -torch.log(torch.tensor(1.0 / 27.0))
logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs = torch.softmax(logits, dim=0)
single_loss = -probs[2].log()
```

Sampling larger, randomly scaled logits (for example `torch.randn(4) * 10`) demonstrates wildly varying losses. The takeaway: ensure initial logits are near zero so the uniform distribution is a valid starting point.

## 5. Diagnosing `tanh` saturation and dead neurons
Histograms of activations and pre-activations tell the story, and the boolean heatmap pinpoints saturated units:

```python
plt.hist(h.view(-1).tolist(), 50)
plt.hist(hpreact.view(-1).tolist(), 50)
plt.plot(torch.arange(-20, 20), torch.tanh(torch.arange(-20, 20)))
plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99, cmap="grey", interpolation="nearest")
```

Most units initially sit at ±1, so gradients vanish because `d/dx tanh(x) = 1 - tanh(x)^2`. A fully white column indicates a dead neuron. The notes highlight that sigmoids suffer similarly, ReLUs can die when they fall into the flat region, and leaky ReLUs avoid that fate.

## 6. Fixing initialization by scaling
Scaling the output weights (`W2 * 0.01`, `b2 = 0`), lightly nudging `W1` and `b1`, and re-running the same training loop removes the “hockey stick” loss curve: the run now starts at the expected 3.29 loss and immediately optimizes. The train/val losses reported (`≈2.11/2.18`) beat the earlier attempt because early iterations are no longer wasted squashing huge logits. Updated histograms confirm that only a few neurons saturate.

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.zeros(vocab_size)
```

## 7. Principled fan-in scaling and gain
Rather than tuning by eye, the notebook runs Gaussian thought experiments to show how variance explodes or vanishes:

```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200) / (10 ** 0.5)
y = x @ w
```

Scaling by `1 / sqrt(fan_in)` keeps the standard deviation near one. For `tanh`, we multiply by the empirically derived gain `5/3` to counteract squashing. PyTorch’s `torch.nn.init.kaiming_normal_` generalizes this idea (`gain = sqrt(2)` for ReLU because half the distribution is zeroed). Applying

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / ((n_embd * block_size) ** 0.5)
```

achieves the same stability as the earlier “magic numbers,” no histogram debugging required.

## 8. Batch Normalization in theory and practice
BatchNorm standardizes each neuron’s pre-activations, then re-scales them with trainable `gamma`/`beta`, so the network can continue to learn arbitrary distributions:

```python
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

hpreact = embcat @ W1 + b1
hpreact = bngain * ((hpreact - hpreact.mean(dim=0, keepdim=True)) /
                    (hpreact.std(dim=0, keepdim=True) + 1e-5)) + bnbias
```

`torch.no_grad()` isolates evaluation-time calls, and the initial calibration pass computes `bnmean`/`bnstd` from the entire training set. Afterwards, running estimates keep inference simple:

```python
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))
bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi
```

Because BatchNorm removes per-layer bias, the notebook drops explicit biases and lets `bnbias` shift the distribution. The notes also call out real-world caveats: BatchNorm couples examples inside a minibatch, introduces jitter that acts like a regularizer, complicates deployment (hence the calibration step), and often motivates LayerNorm or GroupNorm when coupling is undesirable.

## 9. PyTorch-like blocks, ResNet aside, and diagnostics
To mirror `nn.Module`, the notebook implements reusable building blocks and stacks five hidden layers plus a logits head:

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        ...

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
```

This mirrors `nn.Module` where `__init__` wires the layers and `forward()` defines the computation; convolutional layers are the same linear maps applied to sliding patches, which is why ResNet bottlenecks follow the “Weight → Normalization → Non-linearity” motif (Conv/Linear + Norm + ReLU). The stacked network trains with the earlier SGD loop, but every layer’s outputs retain gradients (`layer.out.retain_grad()`) so diagnostics are easy. Visualization snippets cover activation distributions, gradient distributions, parameter-gradient histograms with `grad:data` ratios, and the update-to-data ratio tracked via `ud.append(...)` plus the reference line `plt.plot([0, len(ud)], [-3, -3], 'k')`. Intentionally skipping fan-in scaling makes these plots go haywire (95% saturation, gradients spiking at zero, ratios diverging), which is precisely the point of collecting them. The final softmax weights are shrunk (`layers[-1].weight *= 0.1`) to keep predictions initially uniform, which explains the outlier gradient ratios until the model settles.

## 10. BatchNorm throughout the deep stack
Replacing every “Linear → Tanh” pair with “Linear → BatchNorm → Tanh” (and adding a BatchNorm before the logits) makes the network insensitive to the earlier gain tricks:

```python
layers = [
    Linear(n_embd * block_size, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    ...,
    Linear(n_hidden, vocab_size), BatchNorm1D(vocab_size),
]
with torch.no_grad():
    layers[-1].gamma *= 0.1
```

Because BatchNorm rescales activations, the update ratios shrink; increasing the learning rate to `1.0` brings them back near the `-3` target. Activation histograms now stay at ~0.65 std with minimal saturation, gradients line up across depth, parameter updates progress uniformly, and even random Gaussian weights behave well. The notebook reiterates that you still must retune learning rates whenever you change activation scales.

## 11. Diagnostics playbook
Throughout the notebook, the following checks keep the model honest:

- `split_loss("train")` / `split_loss("val")` under `torch.no_grad()` reveal generalization (baseline numbers settle near 2.11 / 2.18).
- `plt.plot(lossi)` catches the “hockey stick” signature of bad initialization.
- Activation and gradient histograms (plus `plt.imshow(h.abs() > 0.99)`) expose saturation or dead neurons.
- `plt.hist(hpreact.view(-1))` ensures pre-activations stay in the linear region of `tanh`.
- Parameter-gradient histograms and `grad:data` ratios flag layers learning out of sync; the softmax head is intentionally more volatile because its weights were shrunk.
- The update-to-data ratio plot, with the `-3` guideline, double-checks learning-rate calibration—values above it mean steps are too large, values far below it mean learning is painfully slow.
- Sampling via the `torch.multinomial` loop is a qualitative guardrail.

## 12. Where to go next
With careful initialization, principled scaling, BatchNorm, and rich diagnostics, Makemore 3 squeezes as much as possible out of a trigram context (`block_size = 3`). The final notes remind us that context length is now the bottleneck: to push log-probs lower, we must either extend the context window or switch to sequence models such as RNNs and Transformers. Initialization and backprop remain active research topics, so these tools carry forward into the more complex architectures explored next.
