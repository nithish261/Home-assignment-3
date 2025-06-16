# Home-assignment-3

---


## Q1: RNN Text Generation with LSTM

### Description

Trains a character-level text generator using an LSTM-based Recurrent Neural Network (RNN). The model learns to predict the next character in a sequence, trained on a literary text (e.g., "Shakespeare Sonnets").

###  Tasks

* Load a text dataset.
* Preprocess the text into sequences.
* Create an LSTM model using `tensorflow.keras`.
* Train the model and generate text.
* Use temperature scaling to control output randomness.

### Temperature Scaling

Temperature controls the creativity of the generated text:

* Low temperature (e.g., 0.2) → More conservative and predictable text.
* High temperature (e.g., 1.0) → More creative and diverse outputs.

---

## Q2: NLP Preprocessing Pipeline

###  Description

Basic NLP preprocessing including tokenization, stopword removal, and stemming.

### Tools

* `nltk.tokenize.word_tokenize`
* `nltk.corpus.stopwords`
* `nltk.stem.PorterStemmer`

###  Example Input

```text
"NLP techniques are used in virtual assistants like Alexa and Siri."
```

### Outputs

1. **Original Tokens** – Includes all words and punctuation.
2. **Without Stopwords** – Retains only significant words.
3. **Stemmed Words** – Words are reduced to their root form.

---

## Q3: Named Entity Recognition (NER) with spaCy

### Description

Uses `spaCy` to extract and display named entities from a given sentence.

###  Example Sentence

```text
"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."
```

###  Output

For each entity:

* Entity Text (e.g., "Barack Obama")
* Label (e.g., PERSON, DATE)
* Character Span (start, end index)

---

##  Q4: Scaled Dot-Product Attention

###  Description

Implements scaled dot-product attention as used in Transformers.

###  Steps

1. Compute dot product: `Q · Kᵀ`
2. Scale by `sqrt(d)`
3. Apply `softmax`
4. Multiply by `V`

###  Test Input

```python
Q = [[1, 0, 1, 0], [0, 1, 0, 1]]
K = [[1, 0, 1, 0], [0, 1, 0, 1]]
V = [[1, 2, 3, 4], [5, 6, 7, 8]]
```

###  Output

* Attention Weights Matrix
* Final Output after applying attention to V

---

##  Q5: Sentiment Analysis with HuggingFace Transformers

###  Description

Performs sentiment classification using HuggingFace’s Transformers pipeline.

###  Tools

* `transformers.pipeline("sentiment-analysis")`

###  Example Sentence

```text
"Despite the high price, the performance of the new MacBook is outstanding."
```

###  Output

* **Sentiment:** POSITIVE or NEGATIVE
* **Confidence Score:** Float (0–1)

---

## Project Structure

```
.
├── q1_rnn_text_gen.py
├── q2_nlp_preprocessing.py
├── q3_spacy_ner.py
├── q4_scaled_attention.py
├── q5_sentiment_analysis.py
└── README.md
```

---

##  Dependencies

```bash
pip install tensorflow nltk spacy numpy transformers
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

---

