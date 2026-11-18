## NLP stack (use these, don't add others without asking)
- Baseline: scikit-learn (TfidfVectorizer word+char, LogisticRegression/LinearSVC).
- Encoder & perplexity: Hugging Face transformers/datasets (open-weight models only).
- Linguistic/stylometric features: spaCy (en_core_web_sm, de_core_news_sm).
- Error features: language_tool_python (local server only).
- Do not use NLTK, Gensim, Flair, Polyglot, or any closed LLM API.
- Follow COMPETITION.md strictly; fit all transformers on the train fold only.

## Code style: good vs bad

### 1. Named data, not tuple indexing
```python
# BAD
X_s, X_v = views["word"][1], views["word"][2][0]
# GOOD
features = {"word": {"train": X_train_word, "val": X_val_word}}
X_train, X_val = features["word"]["train"], features["word"]["val"]
```

### 2. Configs are data, not lambdas
```python
# BAD
MODELS = {"LogReg": (lambda C, cw: LogisticRegression(C=C, class_weight=cw), [3, 10])}
# GOOD
MODELS = {"LogReg": LogisticRegression}
GRID = {"LogReg": {"C": [3, 10], "class_weight": [None, "balanced"]}}
```

### 3. Flat loops over a config list
```python
# BAD
for f in FEATS:
    for m in MODELS:
        for C in Cs:
            for cw in CWS: ...
# GOOD
configs = [(f, m, p) for f in FEATS for m in MODELS for p in ParameterGrid(GRID[m])]
for feat, model_name, params in tqdm(configs, desc="grid"):
    rows.append(run_one(feat, model_name, params))
```

### 4. No leakage: fit on the train fold only
```python
# BAD
vec = TfidfVectorizer().fit(pd.concat([train.text, val.text, test.text]))
# GOOD
vec = TfidfVectorizer(**TFIDF_PARAMS).fit(train_fold.text)
X_train, X_val, X_test = (vec.transform(d.text) for d in (train_fold, val_fold, test))
```

### 5. Compute returns data, display happens separately
```python
# BAD
for name, clf in fitted.items():
    print(name, classification_report(y_val, clf.predict(X_val)))
# GOOD
results = run_grid()            # returns a DataFrame
results.head(10).round(4)       # next cell: one tidy table
```

### 6. Small functions with a docstring and a single purpose
```python
# BAD: one 80-line cell that loads, vectorizes, trains, plots, and saves
# GOOD
def build_tfidf(train_text, analyzer, ngram_range):
    """Fit a TF-IDF vectorizer on the train fold only and return it."""
    return TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                           sublinear_tf=True, min_df=3).fit(train_text)
```