These are the main commands to control Quarto within the book:

```sh
source .venv/bin/activate

quarto preview
quarto render
```

To install the spacy models (i.e., after recloning), do this:

```sh
source .venv/bin/activate

pip3 install --upgrade pip
python -m ensurepip
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```
