## NLP stack (use these, don't add others without asking)
- Baseline: scikit-learn (TfidfVectorizer word+char, LogisticRegression/LinearSVC).
- Encoder & perplexity: Hugging Face transformers/datasets (open-weight models only).
- Linguistic/stylometric features: spaCy (en_core_web_sm, de_core_news_sm).
- Error features: language_tool_python (local server only).
- Do not use NLTK, Gensim, Flair, Polyglot, or any closed LLM API.
- Follow COMPETITION.md strictly; fit all transformers on the train fold only.