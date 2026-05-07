---
---

## Introduction
Character level language model that predicts the next character given previous characters.

Example: For 'isabella': 
- i likely comes first
- s after i
- a after is
- b after isa, and so on

Representation: `<START>isabella<END>`

---

## 1. Loading the Dataset

```python
with open('names.txt', 'r') as file:
    words = file.read().splitlines()
```
`names.txt` contains around 32000 english names.  

Check dataset statistics:
```python
words[:10]
min(len(word) for word in words)
max(len(word) for word in words)
```
```
['emma', 'olivia', ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia', 'harper', 'evelyn']
2
15
```
---

## 2. Bigram Language Model

**Bigram language model**: Working with two characters at a time. Given a char, predict next character.

**Bigrams**: Two characters in a sequence
    -  `('a', 'b')` : b comes after a in sequence

### Example with a single word
Here we create bigrams out of single word 'emma'.
```python
word = words[0]

zips = zip(word, word[1:])
print('word: ', word)
print(f'bigrams: {*zips}')
```
```
word: emma 
bigrams: ('e', 'm') ('m', 'm') ('m', 'a')
```


### Adding special tokens
Add special tokens to represent start and end of a word.  

for single word 'emma' it looks like this:  
```python
word, ['<S>'] + list(word) + ['<E>']
```
```
('emma', ['<S>', 'e', 'm', 'm', 'a', '<E>'])
```


### Extracting bigrams from multiple words
Extract bigrams from first 3 words of dataset:  

```python
# Two consecutive characters
for word in words[:3]:
    chs = ['<S>'] + list(word) + ['<E>']

    print(f'{word}: ', end='')
    for c1, c2 in zip(chs, chs[1:]):
        print((c1, c2), end=',')
    print()
```
```
emma: ('<S>', 'e'),('e', 'm'),('m', 'm'),('m', 'a'),('a', '<E>'),
olivia: ('<S>', 'o'),('o', 'l'),('l', 'i'),('i', 'v'),('v', 'i'),('i', 'a'),('a', '<E>'),
ava: ('<S>', 'a'),('a', 'v'),('v', 'a'),('a', '<E>'),
```

**Note**: `zip` halts if any list is shorter than the other.

---

## 3. Counting Bigrams

Simple way to learn bigram model is to count number of times bigrams occur in training set.  

### Count bigrams for first 3 words
Extract bigrams from first 3 words and count frequency of each one:

```python
b = {}

for word in words[:3]:
    chs = ['<S>'] + list(word) + ['<E>']
    for c1, c2 in zip(chs, chs[1:]):
        bigram = (c1, c2)
        b[bigram] = b.get(bigram, 0) + 1
```

### Count bigrams for all words
```python
# Now lets do this for all the words
b = {}

for word in words:
    chs = ['<S>'] + list(word) + ['<E>']
    for c1, c2 in zip(chs, chs[1:]):
        bigram = (c1, c2)
        b[bigram] = b.get(bigram, 0) + 1

# Get (bigram, counts) tuples
items = b.items()
```
```
dict_items([(('<S>', 'e'), 1), (('e', 'm'), 1), (('m', 'm'), 1), (('m', 'a'), 1), (('a', '<E>'), 3), (('<S>', 'o'), 1), (('o', 'l'), 1), (('l', 'i'), 1), (('i', 'v'), 1), (('v', 'i'), 1), (('i', 'a'), 1), (('<S>', 'a'), 1), (('a', 'v'), 1), (('v', 'a'), 1)])
```


### Sort by counts
Sort bigrams according to their counts 
```python
# sort by count   
# sort by default sorts wrt first element of object, here its bigram

sorted_by_counts_asc = sorted(items, key= lambda kv: kv[1])
sorted_by_counts_desc = sorted(items, key= lambda kv: -kv[1])
```

---

## 4. 2D Count Array with PyTorch

**Goal**: Put counts in a 2D array where:
- Rows are first char
- Columns are second char of bigram
- Each entry is number of counts that they appear

```python
import torch

# 26 letters of alphabet and 2 special tokens <S> and <E> 
# so we need (28, 28) array for above purpose

# Count array
N = torch.zeros((28, 28), dtype=torch.int32)
```

### Creating character lookup tables

We need some lookup table from characters to integers so that we can index into tensor.  
We map each unique character to an integer.  

```python
# Set of all lowercase characters
# This joins all dataset into one big string and set() removes all duplicate characters from that string
# This way we have set of unique characters in dataset.

# sorted list of unique chars in dataset
chars = sorted(list(set(''.join(words))))  

# Lookup table
stoi = {s: i for i, s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
```

### Populate the count array
```python
# Map both chars to their integers
for word in words:
    chs = ['<S>'] + list(word) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```

---

## 5. Visualizing Bigram Counts

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(N)
```

![count_matrix](count_matrix.png)


### Detailed visualization with labels
```python
itos = {i: s for s, i in stoi.items()}

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')

for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='grey')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='grey')

plt.axis('off');
```
![count_matrix_with_labels](count_matrix_with_labels.png)

Each cell represents count of a bigram. eg cell `N[0][0]` gives count of  bigram `(a,a)`.

**Observations**:
- Last row is entirely zero because `<E>` will never come first in bigram
- One column is entirely zero because `<S>` will never come at end in bigram
- Only possible combination is `<S><E>` i.e. a word with no letters

---

## 6. Using Special Token '.'

**Solution**: Change special token to `.` both for starting and ending.

```python
N = torch.zeros((27, 27), dtype=torch.int32)

stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Map both chars to their integers
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='grey')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='grey')

plt.axis('off');
```

![count_matrix_with_special_token](count_matrix_with_special_token.png)

**Observations**:
- First row shows counts of words that start with respective character.
- First column shows count of words that ends with respective character.

---

## 7. Converting Counts to Probabilities

We use the **frequency interpretation of probability**, where the probability of word $w_2$ following $w_1$ is estimated by its relative frequency in the corpus:

$$P(w_2 \mid w_1) = \frac{\text{count}(w_1, w_2)}{\sum_{i} \text{count}(w_1, w_i)}$$

That is, the number of times the bigram $(w_1, w_2)$ appears, divided by the total number of bigrams that start with $w_1$.

In this model, the special token `.` represents a word boundary. So $P(w_2 \mid \texttt{.})$ gives the probability that $w_2$ is a first character of word, or equivalently, the probability of observing the bigram $(\texttt{.},\ w_2)$ which tells how often $w_2$ appears as the first letter of a word in the training data.

For the specific case of $N[0]$ (i.e., $w_1 = \texttt{.}$), the general formula gives:

$$P(w_2 \mid \texttt{.}) = \frac{N[0,\ w_2]}{\displaystyle\sum_{i} N[0,\ i]}$$

where $N[0, w_2]$ is the count of the bigram $(\texttt{.},\ w_2)$: how many times character $w_2$ appears as the **first letter of a word**, and the denominator $\sum_i N[0, i]$ is the total count of all bigrams starting with $\texttt{.}$, i.e. the total number of `(., i)` bigrams int he corpus for all characters `i`.

In code this is exactly:
```python
p = N[0].float()   # numerators: N[0, w_2] for each w_2
p /= p.sum()       # divide by sum_i N[0, i]
```

So `p[i]` $= P(w_2 = i \mid w_1 = \texttt{.})$: the probability of the bigram ('.', i): the probability that the i-th character appears as the first letter of a word.

---

## 8. Sampling from Probability Distribution

### Understanding torch.multinomial

```python
# Deterministic way of creating a torch generator object
h = torch.Generator().manual_seed(2147483647) 

# We use generator object as source of randomness in following function

# Gives 3 random numbers between 0 and 1: modelling probs of 3 indices
p = torch.rand(3, generator=h)   #  [0.7081, 0.3542, 0.1054] 

p = p / p.sum()
p  # [0.6064, 0.3033, 0.0903]
```
multinomial will sample first index 60% of times, second about 30% of time and so on.

```python
# Use torch multinomial to draw 100 samples from above randomly generated p

torch.multinomial(p, num_samples=100, replacement=True, generator=h)
```
```
tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
        0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0,
        0, 1, 1, 1])
```
Here, first index is sampled about 60 times, second one 30 and third one 9 times (approximately). 

### Sampling first character

Now we sample from our first row same as done above.

```python
# Convert these counts to probabilities
p = N[0].float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647) 
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
ix, itos[ix]
```
```
(3, 'c')
```
**Note**: This result different than one in lecture video, probably due to change in library itself.


### Sampling next character

Now that our first sampled char is 'c', we go to row corresonding to 'c', .i.e. row at index 3 and sample next character.

```python
p = N[3].float()
p /= p.sum()

ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
ix, itos[ix]
```
```
(5, 'e')
```
We continue generate next characters until end token '.' is generated.

---

## 9. Generating Words with Loop

**Algorithm**:
1. initialize `ix = 0`, which corresponds to the special token `'.'`, representing the start of a word.
2. Then in loop:
   - Sample the next character from row `ix` of the probability matrix, i.e., draw from $P(w_2 \mid w_1 = ix)$
   - Set `ix` to the sampled character index.
- Repeat until `ix = 0` is sampled again, signaling the end of the word


At each step, the current character `ix` acts as the first character, and we sample the *next* character from its corresponding row.  
Then next character is set as current character and loop continues.  

When `ix = 0` (`'.'`) is sampled, it marks a word boundary and the loop terminates.


Below we generate 10 words
```python
g = torch.Generator().manual_seed(2147483647) 

for i in range(10):
    out = []
    ix = 0
    
    while True:
        p = N[ix].float()
        p /= p.sum()

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print(''.join(out))
```
```
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
da.
staiyaubrtthrigotai.
moliellavo.
ke.
teda.
```

**Result**: As you can see, bigrams are terrible and we should do better. But bigrams are still better than untrained model.  
See below section for words generated by untrained model (random unirform sampling).

---

## 10. Comparison with Uniform Sampling

Following model samples uniformly from 27 characters:

```python
g = torch.Generator().manual_seed(2147483647) 

for i in range(10):
    out = []
    ix = 0
    
    while True:
        p = torch.ones(27)
        p /= 27  # Uniform probability

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print(''.join(out))
```
```
cexzm.
zoglkurkicqzktyhwmvmzimjttainrlkfukzkktda.
sfcxvpubjtbhrmgotzx.
iczixqctvujkwptedogkkjemkmmsidguenkbvgynywftbspmhwcivgbvtahlvsu.
dsdxxblnwglhpyiw.
igwnjwrpfdwipkwzkm.
desu.
firmt.
gbiksjbquabsvoth.
kuysxqevhcmrbxmcwyhrrjenvxmvpfkmwmghfvjzxobomysox.
```

This is garbage. So bigrams are one step better than this, but still are terrible.

---

## 11. Broadcasting and Efficient Normalization

Don't normalize every row (dividing cells by their row sum) every time, instead compute probabilities at once. So that every row contains prob distribution over 27 words, given previous word: Calculate matrix `P` once then use it for generation.

### Understanding torch.sum with dimensions

`P.sum(input, dim, keepdim=True)`:
- When given dim, sum is performed **across** that dim
- `dim=0` (rows in `[27, 27]`): **sum is performed across rows i.e. Each column is summed across all rows.**
  - Vertical sum resulting in `[1, 27]` row vector
- `dim=1` (columns in `[27, 27]`): **sum is performed across columns i.e. Each row is summed across all columns.**
  - Horizontal sum resulting in `[27, 1]` column vector
- `keepdim=True`: preserve reduced dimension(s) with size 1 
    - `keepdim=False`: result is `(27,)` 
    - `keepdim=True`: result is `(1, 27)` or `(27, 1)` depending on `dim`.

```python
P = N.float()
P /= P.sum(dim=1, keepdim=True)
P.shape  # [27, 27]

P[0].sum()  # Should be 1.0
```
If `P` is `(m, n)` matrix, then  
`P.sum(dim=0, keepdim=False)` gives *sum across rows*: Each column collapsed into one number by addition, so output shape would be `(n,)` vector.

___

### Broadcasting Rules

Broadcasting has rules in pytorch. Visit docs for more information.

Consider matrix with shape (27, 27) and vector with shape (27,), and we divide them.  
Note division is boradcasting supported operation.  
Here is how broadcasting mechanism will play out.

**Rule 1**: Align all dimensions from the right:  

```plaintext
    [27, 27]
    [27]
→
    [27, 27]
    [    27]
```

**Rule 2**: Iterate over all dimensions (columns) starting from right to left. Each dimension must be either: equal to other, or one of them is 1, or one of them does not exist.
Intenrally boradcasting will create dimension where it does not exist:
```
→
    [27, 27]
    [1,  27]
```
**Rule 3**: Broadcast dimension with 1 to match dimension of other matrix 

Broadcasting copies `[1, 27]` row vector 27 times, stacking as rows i.e. along first dimension, to make it `(27, 27)` matrix, where first dimension is now matched for both matrices.  
```
→
    [27, 27]
    [27, 27]
```

Now it does element-wise division.


**How `keepdim=False` can causes issues** 

`keepdim=False` does not preserve which dimension was summed across.  
Boradcasting rules can produce unexpected results due to this.  

Consider following example:

To normalize counts matrix N, we want each cell of N to be divided by sum of its row elements.  
The row sum for entire matrix is calculated using `N.sum(dim=1)`,  
Lets call this vector `row_sum`.  
    
    row_sum = N.sum(dim=1, keepdims=False)    -> (27,)  # Row sum vector: first element of this vector is sum of elements of first row and so on  
    P = N / row_sum    -> (27, 27) / (27,)

    Boradcasting applied: 
    1. Align to right 
        [27, 27]
        [    27]
    2. Internally dimension of size 1 is created if not exist already.
        [27, 27]
        [1,  27]

This resulted in `row_sum` vector to be a row vector `(1, 27)`: `[sum_of_row1 sum_of_row2 sum_of_row3 ... sum_of_row_m]`  

    3. Broadcast this vector into first dimension 
        [27,  27]
        [27,  27]

Here `row_sum` row vector is copied 27 times and staced as 27 rows, resulting in (27, 27) matrix.  
        
row_sum is now this matrix
        $$ 
        \begin{pmatrix}
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m \\
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m \\
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m
        \end{pmatrix}
        $$ copied 27 times. 


**Problem**: Now `row_sum` is a matrix where along columns we have sum for of single row.
- We want to divide a row element by sum of all elements of that row
- for this we need `row_matrix` where each row has its row sum only, this way we can do elementwise division.
- Above broadcasting creates a matrix that has row sums along columns and not rows
- So we're dividing **first entry of each row by sum of first row**, second column by sum of second row, and so on
- i.e. we are normalizing the columns instead of rows

What is happening:
        $$
        \begin{pmatrix}
        N_{11} & N_{12} & N_{13} & \cdots & N_{1,27} \\
        N_{21} & N_{22} & N_{23} & \cdots & N_{2,27} \\
        N_{31} & N_{32} & N_{33} & \cdots & N_{3,27} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        N_{27,1} & N_{27,2} & N_{27,3} & \cdots & N_{27,27}
        \end{pmatrix}
        \div
        \begin{pmatrix}
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m \\
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m \\
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        \text{sum\_of\_row}_1 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_m
        \end{pmatrix}
        $$
  
- This is not our desired behavior

What we want: 
    $$
    \begin{pmatrix}
    N_{11} & N_{12} & N_{13} & \cdots & N_{1,27} \\
    N_{21} & N_{22} & N_{23} & \cdots & N_{2,27} \\
    N_{31} & N_{32} & N_{33} & \cdots & N_{3,27} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    N_{27,1} & N_{27,2} & N_{27,3} & \cdots & N_{27,27}
    \end{pmatrix}
    \div
    \begin{pmatrix}
    \text{sum\_of\_row}_1 & \text{sum\_of\_row}_1 & \text{sum\_of\_row}_1 & \cdots & \text{sum\_of\_row}_1 \\
    \text{sum\_of\_row}_2 & \text{sum\_of\_row}_2 & \text{sum\_of\_row}_2 & \cdots & \text{sum\_of\_row}_2 \\
    \text{sum\_of\_row}_3 & \text{sum\_of\_row}_3 & \text{sum\_of\_row}_3 & \cdots & \text{sum\_of\_row}_3 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \text{sum\_of\_row}_{27} & \text{sum\_of\_row}_{27} & \text{sum\_of\_row}_{27} & \cdots & \text{sum\_of\_row}_{27}
    \end{pmatrix}
    $$

This `row_sum` matrix could have been resulted with `keepdims=True`:  

    row_sum = N.sum(dim=1, keepdims=True)    -> (27, 1)  # Now this is a column vector, where first element is first row sum, and so on. 
    P = N / row_sum    -> (27, 27) / (27, 1)

    Boradcasting applied: 
    1. Align to right 
        [27, 27]
        [27,  1]
No need of creating extra dimension,

    2. Copy column vector 27 times, stacked as columns 
        [27,  27]
        [27,  27]

This results in row_sum matrix that we desired above, with each row containing only its row sum.  


**Lesson**: Have respect for broadcasting, check your work, understand how it works under the hood, and make sure broadcasting is working in the direction that you want, otherwise you'll introduce very subtle and hard to detect bugs.

___

### Using probability matrix P for sampling

```python
g = torch.Generator().manual_seed(2147483647) 

for i in range(10):
    out = []
    ix = 0
    
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print(''.join(out))
```
```
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
da.
staiyaubrtthrigotai.
moliellavo.
ke.
teda.
```
We get exact same results as before without having to normalize at every iteration.

**Note**: 
- `P = P / P.sum()` creates a new tensor `P`
- `P /= P.sum()` operates inplace

---

## 12. Model Summary

So, now we have trained a bigram model by counting frequency of pairs and then normalizing counts to get probability distribution.
Elements of `P` are really the parameters of our bigram model, summarizing statistics of bigrams.

---

## 13. Evaluating Quality of Model

### Using Negative Log Likelihood

Now we need to summarize quality of this trained model into a single number. i.e. how good model is in predicting the training set.

One example is **training loss** which tells us how model did in training against dataset.

Lets look at probabilities of some bigrams:
```python
for word in words[:3]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        print(f'{ch1}{ch2}: {prob:.4f}')
```
```
.e: 0.0478
em: 0.0377
mm: 0.0253
ma: 0.3899
a.: 0.1960
.o: 0.0123
ol: 0.0780
li: 0.1777
iv: 0.0152
vi: 0.3541
ia: 0.1381
a.: 0.1960
.a: 0.1377
av: 0.0246
va: 0.2495
a.: 0.1960
```

**Interpretation**:
- These are the probs that model assigned to each bigram in dataset
- If every bigram was equally likely, then these probs would have been `1/27 ≈ 0.0370`, roughly 4%
- If any prob is above 4% means we have learnt something useful from bigram statistic
- Model has assigned pretty good probs for what's in training set (some are 4%, some are 17%, 35%, 40%)
- If you had a very good model, these probs for each bigram in train set would be near 1

### Maximum Likelihood Estimation
To summarize these probabilities into a single measure of model quality, in literature, we use **Maximum Likelihood Estimation**.  

The likelihood is simply the **product of all predicted probabilities for the correct labels**:

$$\mathcal{L} = \prod_{i=1}^{N} P(y_i \mid x_i)$$

Where:
- $N$ = number of samples in the dataset
- $x_i$ = input for sample $i$
- $y_i$ = true label for sample $i$
- $P(y_i \mid x_i)$ = probability the model assigns to the correct label 

This is probability of occurence of all correct labels for all bigrams.     
$P(y_i \mid x_i)$ is probability of bigram $(x_i, y_i)$.  
It is assumed that every bigram $(x_i, y_i)$ is independent of other, so probability of their simultaneous occurence (joint probability) is given by product of their individual probabilities, by *independence assumption*.

Likelihood tells us probability of entire dataset assigned by the trained model.
Product of these probs should be as high as possible to have a good model.

---
For convenience we use **log of probs**:  
taking the log turns the product into a sum:

$$\log \mathcal{L} = \sum_{i=1}^{N} \log P(y_i \mid x_i)$$
___

- Here log is natural log. 
- `log(1) = 0`
- As we go below 1, log falls to negative values till `log(0) = -inf`
- If all truth label probs are near 1, then log likelihood would be near 0.
- If probs are near 0, log likelihood would be more negative.

We have to maximize log likelihood toward 0 (its upper bound), to get our probs to near 1.  
But we want to minimize the loss. So we use **negative log likelihood**.
    
    negative log likelihood = - (log likelihood)

Minimizing **negative log likelihood** is equiavalent to maximizing log likelihood.  

**Negative Log Likelihood (NLL)** loss is:

$$\text{NLL} = -\sum_{i=1}^{N} \log P(y_i \mid x_i)$$

- When probs go from 0 to 1:
    - log likelihood goes from -inf to 0
    - **-log likelihood goes from +inf to 0** (what we want for loss)

Thus, minimizing negative log likelihood (NLL) cause log likelihood to go to 0 which in turn causes all truth label probs to go to 1.  


### Computing Negative Log Likelihood

```python
log_likelihood = 0.0
for word in words[:3]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        print(f'{ch1}{ch2}: {prob:.4f}, {log_prob}')
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
```
```
.e: 0.0478, -3.0408456325531006
em: 0.0377, -3.2793259620666504
mm: 0.0253, -3.6772043704986572
ma: 0.3899, -0.9417552351951599
a.: 0.1960, -1.629860520362854
.o: 0.0123, -4.3981709480285645
ol: 0.0780, -2.550807476043701
li: 0.1777, -1.7277942895889282
iv: 0.0152, -4.186665058135986
vi: 0.3541, -1.0382848978042603
ia: 0.1381, -1.9795759916305542
a.: 0.1960, -1.629860520362854
.a: 0.1377, -1.9828919172286987
av: 0.0246, -3.7044942378997803
va: 0.2495, -1.3882395029067993
a.: 0.1960, -1.629860520362854
log_likeliood=tensor(-38.7856)
nll=tensor(38.7856)
```

**Why NLL is a good loss function**:
- It's always ≥ 0
- When probs are near 1, it's near to 0
- When probs are away from 1, it increases away from 0
- Higher the NLL, worse the predictions are

**For convenience, we use average negative log likelihood.**
NLL averaged over samples:

$$\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i \mid x_i)$$


### Average Negative Log Likelihood

```python
# Average log likelihood
log_likelihood = 0.0
n = 0

for word in words[:3]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f}, {log_prob}')
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll/n=}')
```
```
.e: 0.0478, -3.0408456325531006
em: 0.0377, -3.2793259620666504
mm: 0.0253, -3.6772043704986572
ma: 0.3899, -0.9417552351951599
a.: 0.1960, -1.629860520362854
.o: 0.0123, -4.3981709480285645
ol: 0.0780, -2.550807476043701
li: 0.1777, -1.7277942895889282
iv: 0.0152, -4.186665058135986
vi: 0.3541, -1.0382848978042603
ia: 0.1381, -1.9795759916305542
a.: 0.1960, -1.629860520362854
.a: 0.1377, -1.9828919172286987
av: 0.0246, -3.7044942378997803
va: 0.2495, -1.3882395029067993
a.: 0.1960, -1.629860520362854
log_likeliood=tensor(-38.7856)
nll/n=tensor(2.4241)
```

**Thus we use average negative log likelihood as our loss function.**  
Our aim is to minimize this loss to get high quality model.

### Optimization Goal

**GOAL**: Maximize likelihood of the data wrt model parameters (statistical modelling)

(Later these parameters (counts here) will be calculated by a NN and we want to tune these parameters to maximize likelihood of training data)

**Equivalences**:
- Maximize likelihood
- ≡ Maximize log likelihood (because log is a monotonic function, maxizing products of probs and maximing sum of log of probs are the same thing).
- ≡ Minimize negative log likelihood
- ≡ Minimize average negative log likelihood

### Loss on Entire Training Set

```python
# Average log likelihood for entire training set
log_likelihood = 0.0
n = 0

for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll/n=}')
```
```
log_likeliood=tensor(-559891.7500)
nll/n=tensor(2.4541)
```

### Testing on New Data

```python
# Test on a name not in dataset
log_likelihood = 0.0
n = 0

for word in ["andrejq"]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll/n=}')
```
```
log_likeliood=tensor(-inf)
nll/n=tensor(inf)
```

---

## 14. Laplace Smoothing

**Problem**: If any count is 0:
- `p((ai, aj)) = 0`
- `log(p(ai, aj)) = -inf`
- `-log(p(ai, aj)) = inf`
- `NLL = AVG(all individual nll) = inf`
- This means entire sequence can have infinite loss due to single bigram prob being 0 and its nll being `inf`.

**Solution**: Add 1 to every count so no count is 0. This is called as Laplace Smoothing.

Build P with laplace smoothing:  
```python
P = (N+1).float()  # Add one here for smoothing
P /= P.sum(dim=1, keepdim=True)
```

Test with smoothing:
```python
# Average log likelihood for entire training set
log_likelihood = 0.0
n = 0

for word in ['andndrejq']:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
    
print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll/n=}')
```
```
log_likeliood=tensor(-36.2776)
nll/n=tensor(3.6278)
```
Now the loss is not inf as it was before.  

---

## 15. Neural Network Approach

So we trained a bigram character level model by means of counting, normalizing counts to get probability dist and sampling from that dist to generate words, evaluated model using negative log likelihood.

Now we frame character level bigram model in **NN framework**: 
*It inputs one char and outputs prob dist for next character.*

- Inputs one char
- Outputs prob dist for next character
- We have bigrams as training set, so we know next character given first, we can evaluate model based on this
- NN outputs prob dist over next char, we have target labels and a loss function: nll.
- Model should assign high prob to next char i.e. loss should be low

### Creating Training Set

Let's first create training set of all bigrams `(x, y)` from first word:
```plaintext
(x, y)
x: input (int)
y: target (int)

Given x, predict y
```

```python
# create training set of bigrams (x, y)

xs, ys = [], []

for word in words[:1]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xs, ys
```
```
(tensor([0,  5, 13, 13,  1]), tensor([5, 13, 13,  1,  0]))
```
These are formed from bigrams: `[(0, 5), (5, 13), (13, 13), (13, 1), (1, 0)]` where each bigram is in format `(x, y)`.  
`xs: [0,  5, 13, 13,  1]` and `ys=[5, 13, 13,  1,  0]`.  

**Note**: xs and ys are index of characters. Indexes are integers.  

---

## 16. One-Hot Encoding

Its not recommended to input an integer to NN.

**Problem with integers as input**:
- We multiply them with weights which are floats, so they should be float
- Integers imply numerical relationship between indexes
- If 'a' index is 1 and 'b' index is 2, numerically 'b' is greater than 'a'
- Character 'm' (index 13) is not "halfway" between 'a' (index 1) and 'y' (index 25)
- All characters should be treated as equally distinct from each other
- Characters are categorical data, not continuous

**Solution**: Common way of encoding integers is **one-hot encoding**.
One hot encoding:  
    - Vector of size total characters possible, here 27 
    - 0 everywhere except at index of character  

eg, one hot vector of size 27 for character `c` which has index 3 is:  
$$[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$$

Visualize `xs` as one hot vectors:   
`xs=[0,  5, 13, 13,  1]`
```python
import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27)
xenc.shape  # torch.Size([5, 27])

plt.imshow(xenc)
```
![One hot visualization](one_hot_examples.png)


**Interpretation**:

- Yellow squares have value 1, all others have value 0.
- 5 examples (inputs) encoded as 5 row vectors
- We will feed each such example to NN
- We want input to be floats that are able to take various real values (ints can't)

```python
xenc.dtype  # torch.int64

# Cast one hot vectors to float dtype
xenc = F.one_hot(xs, num_classes=27).float()
print(xenc.shape, xenc.dtype)
plt.imshow(xenc)
```
```
torch.Size([5, 27]) torch.float32
```

![One hot visualization](one_hot_examples_2.png)


---

## 17. Neural Network Forward Pass

### Single Neuron

```python
# initialize Weights for a single neuron that will input above vectors 
W = torch.randn((27, 1))
xenc @ W  # Output: (5, 1)
```
```
tensor([[-1.5376],
        [-0.1570],
        [ 1.0750],
        [ 1.0750],
        [-1.7193]])
```

`@` is matrix multiplication operator in pytorch.

Here we fed all 5 inputs to this neuron and it produced its activations `(5, 1)`.

### 27 Neurons (Full Layer)

```python
# neurons stacked as columns
W = torch.randn((27, 27))
xenc @ W  # Output: (5, 27)
```

We can efficiently calculate activations by passing inputs stacked as rows as a batch and multiplying them with weights of neurons stacked as columns.

### Network Architecture

Our NN for now will be:
- 27 dimensional input
- 27 neurons in first linear layer which outputs prob of next char

We will treat 27 numbers that output as **log of counts** (not integer counts because NN should not output an int):
- Log of counts are also called **logits**

### From Logits to Probabilities

So how we interpret 27 output numbers: they are log counts.  
Exponentiate log counts and you get counts.

**Exponential function**:  $e^x$ or $\exp(x)$
- x: Negative numbers → output below 1 but greater than 0: `(0, 1)`
- x: Positive numbers → >1 up to +inf: `(1, +inf)`

So exp are good candidates for counts: never below 0 and can take on various values, depending on settings of W.

```python
(xenc @ W).exp()
```
```
tensor([[2.5169, 0.9381, 0.2880, 1.6197, 2.8216, 1.0193, 2.0663, 0.5789, 0.7802,
         0.4641, 2.9903, 0.2530, 1.8502, 0.6355, 3.8250, 3.4950, 0.3467, 2.6788,
         7.2475, 1.3295, 1.8077, 2.2006, 0.3396, 3.1215, 0.1890, 5.2692, 1.9253],
        [0.5295, 1.1082, 0.6860, 2.8803, 0.8538, 0.5382, 0.5677, 1.1434, 0.4833,
         1.9150, 0.2720, 4.6556, 3.8992, 2.1483, 3.1176, 0.9707, 1.8023, 2.1434,
         3.5181, 2.9053, 0.1588, 0.7161, 0.3570, 1.8890, 0.8244, 0.5981, 2.9646],
        [8.6271, 0.1702, 0.6642, 3.8820, 2.7708, 0.4509, 2.1952, 0.4544, 0.7953,
         0.5790, 0.3022, 0.4205, 1.7348, 0.6330, 3.1612, 0.5826, 1.1090, 0.4046,
         2.9894, 2.5377, 3.5922, 3.0635, 1.2510, 0.2189, 0.3091, 0.1984, 1.7693],
        [8.6271, 0.1702, 0.6642, 3.8820, 2.7708, 0.4509, 2.1952, 0.4544, 0.7953,
         0.5790, 0.3022, 0.4205, 1.7348, 0.6330, 3.1612, 0.5826, 1.1090, 0.4046,
         2.9894, 2.5377, 3.5922, 3.0635, 1.2510, 0.2189, 0.3091, 0.1984, 1.7693],
        [0.5107, 0.3904, 0.6115, 4.1294, 0.2303, 1.6448, 8.1907, 1.1071, 3.1120,
         2.0898, 0.4168, 0.2154, 1.4509, 1.6455, 0.9134, 0.3969, 1.7598, 0.9947,
         0.2282, 2.5112, 0.3759, 0.3582, 2.0293, 1.5503, 1.1108, 0.8028, 0.2594]])
```

These numbers can be interpreted as equivalent of counts.  

### Complete Transformation

```python
logits = xenc @ W  # log-counts
counts = logits.exp()  # counts equivalent to N
probs = counts / counts.sum(dim=1, keepdim=True)

probs.shape  # (5, 27)
probs[1].sum()  # Should be 1.0
```

So for every one of 5 examples we have a row that came out of a NN, and with above transformations, we made sure that outputs can be interpreted as probabilities.

All above operations are differentiable that can be backpropagated.

**Process**:
We fed `.` by
- getting its index
- One hot encoded the index
- Fed to NN
- Output prob dist after transformations

These probs are NN's assignment of prob for next character.

We now want to tune W such that good probs are output.

---

## 18. Training the Neural Network

### Setup

```python
# create training set 
xs, ys = [], []

for word in words[:1]:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# Randomly initialize 27 neurons' weights, each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

xenc = F.one_hot(xs, num_classes=27).float()  # input to network one hot encoded
logits = xenc @ W  # predict log-counts

counts = logits.exp()  # counts equivalent to N
probs = counts / counts.sum(dim=1, keepdim=True)  # Probabilities for next character

# Last two lines here are together called a 'softmax'
```

### Softmax

**Softmax**: Takes logits, exponentiates them and normalizes them.  
It takes outputs of NN which can be positive or negative and outputs probability distribution i.e. something that sums to one and contains only positive numbers, just like probabilities.

### Computing Loss
Compute loss for 5 examples:  

```python
nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram:
    x = xs[i].item()  # input character index
    y = ys[i].item()  # label character index
    print('--------')
    print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
    print('input to the neural net:', x)
    print('output probabilities from the neural net:', probs[i])
    print('label (actual next character):', y)
    p = probs[i, y]
    print('probability assigned by the net to the correct character:', p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())
```
```
--------
bigram example 1: .e (indexes 0,5)
input to the neural net: 0
output probabilities from the neural net: tensor([0.0607, 0.0100, 0.0123, 0.0042, 0.0168, 0.0123, 0.0027, 0.0232, 0.0137,
        0.0313, 0.0079, 0.0278, 0.0091, 0.0082, 0.0500, 0.2378, 0.0603, 0.0025,
        0.0249, 0.0055, 0.0339, 0.0109, 0.0029, 0.0198, 0.0118, 0.1537, 0.1459])
label (actual next character): 5
probability assigned by the net to the the correct character: 0.01228625513613224
log likelihood: -4.399273872375488
negative log likelihood: 4.399273872375488
--------
bigram example 2: em (indexes 5,13)
input to the neural net: 5
output probabilities from the neural net: tensor([0.0290, 0.0796, 0.0248, 0.0521, 0.1989, 0.0289, 0.0094, 0.0335, 0.0097,
        0.0301, 0.0702, 0.0228, 0.0115, 0.0181, 0.0108, 0.0315, 0.0291, 0.0045,
        0.0916, 0.0215, 0.0486, 0.0300, 0.0501, 0.0027, 0.0118, 0.0022, 0.0472])
label (actual next character): 13
probability assigned by the net to the the correct character: 0.018050700426101685
log likelihood: -4.014570713043213
negative log likelihood: 4.014570713043213
--------
bigram example 3: mm (indexes 13,13)
input to the neural net: 13
output probabilities from the neural net: tensor([0.0312, 0.0737, 0.0484, 0.0333, 0.0674, 0.0200, 0.0263, 0.0249, 0.1226,
        0.0164, 0.0075, 0.0789, 0.0131, 0.0267, 0.0147, 0.0112, 0.0585, 0.0121,
        0.0650, 0.0058, 0.0208, 0.0078, 0.0133, 0.0203, 0.1204, 0.0469, 0.0126])
label (actual next character): 13
probability assigned by the net to the the correct character: 0.026691533625125885
log likelihood: -3.623408794403076
negative log likelihood: 3.623408794403076
--------
bigram example 4: ma (indexes 13,1)
input to the neural net: 13
output probabilities from the neural net: tensor([0.0312, 0.0737, 0.0484, 0.0333, 0.0674, 0.0200, 0.0263, 0.0249, 0.1226,
        0.0164, 0.0075, 0.0789, 0.0131, 0.0267, 0.0147, 0.0112, 0.0585, 0.0121,
        0.0650, 0.0058, 0.0208, 0.0078, 0.0133, 0.0203, 0.1204, 0.0469, 0.0126])
label (actual next character): 1
probability assigned by the net to the the correct character: 0.07367686182260513
log likelihood: -2.6080665588378906
negative log likelihood: 2.6080665588378906
--------
bigram example 5: a. (indexes 1,0)
input to the neural net: 1
output probabilities from the neural net: tensor([0.0150, 0.0086, 0.0396, 0.0100, 0.0606, 0.0308, 0.1084, 0.0131, 0.0125,
        0.0048, 0.1024, 0.0086, 0.0988, 0.0112, 0.0232, 0.0207, 0.0408, 0.0078,
        0.0899, 0.0531, 0.0463, 0.0309, 0.0051, 0.0329, 0.0654, 0.0503, 0.0091])
label (actual next character): 0
probability assigned by the net to the the correct character: 0.014977526850998402
log likelihood: -4.201204299926758
negative log likelihood: 4.201204299926758
=========
average negative log likelihood, i.e. loss = 3.7693049907684326  
```

This is not a very good setting of W, as our loss (average negative log likelihood) is much higher than 0.  
This loss is made up of differentiable functions, so we can minimize the loss by tuning W parameters.

---

## 19. Gradient-Based Optimization

### Efficient Loss Computation

We need probs of truth labels to calculate loss:  

```python
# Probs required to calculate loss
probs[0, 5], probs[1, 13], probs[2, 13], probs[3, 1], probs[4, 0]

# Better way to index for this use case
probs[torch.arange(5), ys]
```
```
tensor([0.0123, 0.0181, 0.0267, 0.0737, 0.0150])
```
These are probs that NN assigns to correct next character.

```python
# AVG NLL Loss
loss = -probs[torch.arange(5), ys].log().mean()
```

### Training Loop Setup

```python
# Randomly initialize 27 neurons' weights, each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)  # By default requires_grad is False
```

### Forward and Backward Pass

```python
# Forward pass
xenc = F.one_hot(xs, num_classes=27).float()  # input to network one hot encoded
logits = xenc @ W  # predict log-counts
counts = logits.exp()  # counts equivalent to N
probs = counts / counts.sum(dim=1, keepdim=True)  # Probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean()

# Backward pass
# Make sure all gradients are reset to 0
# Setting grads to None is efficient and interpreted by pytorch as lack of gradient and is same as 0s  

W.grad = None 
loss.backward()

# Update
W.data += -0.1 * W.grad
```

**Note**: Having low loss means network is assigning high probs to correct targets.

---

## 20. Full Training on Dataset

### Create Full Dataset

```python
# create the dataset 
xs, ys = [], []

for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('Number of examples:', num)

# Initialize the network
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```
```
Number of examples:  228146
```


### Gradient Descent

```python
# gradient descent for 100 epochs
for k in range(100):
    
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float()  # input to network one hot encoded
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts equivalent to N
    probs = counts / counts.sum(dim=1, keepdim=True)  # Probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    # backward pass
    W.grad = None 
    loss.backward()

    # update
    W.data += -50 * W.grad

print(loss.item())
```
```
2.4901304244995117
```

**Results**:
- Our loss at starting when we did with counting was around 2.47, roughly 2.45 before smoothing
- So here we have achieved the same performance with gradient based optimization

**Comparison**:
- Counting was straightforward and fast for this problem, we were able to maintain probs in a table
- But NN is flexible approach

**Future improvements**:
- What we can do now is to complexify the NN by *feeding multiple previous characters* into increasingly complex neural nets
- But output of NN will always just be logits, which will go through exact same transformation as above
- The only thing that will change is how we do forward pass, everything else remains same

---

## 21. Neural Network vs Counting Approach

### Scalability

If we are taking multiple previous characters, then it's not possible to keep counts table for every combination, this is unscalable approach.

NN approach on other hand is scalable and we can improve on over time.

### Mathematical Equivalence

```python
logits = xenc @ W
```

Multiplying one hot vector of say 5th character, with W, plucks out 5th row of W, because of how matrix multiplication works.

So `logits` just become 5th row of W.

**In counting approach**:
- We had first char say 5th one
- Then we would go to 5th row of `N` which then gave us prob dist for next char
- So first char was used as a lookup into matrix `N`

**Similar thing is happening in NN**:
- We take index, say 5, encode it one hot and multiply by W
- So logits become appropriate row (here 5th) of W
- Which are then exponentiated into counts and normalized into probability, similar to prob dist for next char we got in counting approach.

**Conclusion**: `W.exp()` at end of optimization is same as `N` array of counts.
- `N` was filled by counting
- `W` was initialized randomly and loss guided us to arrive at same array as `N`

---

## 22. Regularization

### Smoothing Equivalence

In smoothing, if we add `10000` to every count, where max count was around 900, then every count will approximately look the same (min 10000, max 10900) and upon normalization we will have nearly same prob for each character, i.e. we would get a uniform distribution.

Same thing can happen in NN approach:
- `W` initialized to all 0s
- `logits` become all 0s
- `counts = logits.exp()` become all 1s
- `probs = count/count.sum(1, keepdim=True)` become all uniform.

Having weights near 0 during training cause model to output near uniform distribution.  

Due to optimization algorithm, model try to maximize probability of training truth label, this can result in overfitting to training data.  
So incentivizing `W` to be near 0 (not exactly 0) during training push model towards uniform distribution, smoothing output probability distribution and prevents peaky predictions.  
This is same effect as laplace smoothing.  
More you incentivize this in loss function, more smooth dist you achieve.  
This is called **regularization**

### Regularization Loss

**Regularization**: where we can add small component to loss called regularization loss.  
This is done by adding something like `(W**2).mean()` to loss function.

- You achieve 0 loss if `W` is exactly a 0 matrix
- But if `W` has non-zero numbers then you accumulate loss
- You can choose regularization parameter which decides how much regularization affects the loss
- This component tries to make all w's be 0 in optimization

So in optimization with regularization:
- W wants to be 0 -> Probs want to be uniform  -> But also match up your training data

**Regularization parameter is similar to addition factor of count in Laplace smoothing.**

We dont use regularization here.

---

## 23. Sampling from Neural Network
Sample from above trained NN 

```python
# Finally sample from the model
g = torch.Generator().manual_seed(2147483647) 

for i in range(5):
    out = []
    ix = 0
    
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float() 
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts equivalent to N
        p = counts / counts.sum(dim=1, keepdim=True)  # Probabilities for next character

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])

        if ix == 0:
            break

    print(''.join(out))
```
```
cexze.
momasurailezityha.
konimittain.
llayn.
ka.
```

**Result**: Thus we got same samples as bigram counting models.  
So these are fundamentally same models but we came at it in different way and they have different interpretations.

---


## Summary

This notebook covered:
1. **Bigram Language Model**: Building character-level model using counting
2. **Probability Distributions**: Converting counts to probabilities
3. **Sampling**: Generating new names from the model
4. **Evaluation**: Using negative log likelihood as loss function
5. **Neural Network Approach**: Reimplementing bigrams using gradient descent
6. **One-Hot Encoding**: Proper way to feed categorical data to neural networks
7. **Softmax**: Converting logits to probabilities
8. **Regularization**: Smoothing probabilities and preventing overfitting
9. **Equivalence**: Understanding how counting and NN approaches are fundamentally the same

The key insight is that while counting is straightforward for bigrams, the neural network approach is more scalable and can be extended to handle longer context (trigrams, n-grams, etc.) and more complex architectures.
