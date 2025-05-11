# 01-intro module

This readme complements the main [lesson README.md](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/01-intro/README.md)

Useful [FAQ section](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit?tab=t.0#heading=h.5z1uyw9hpgkf)

## Development environment

I have used `uv` to set up the environment. [Here are the instructions](https://docs.astral.sh/uv/getting-started/installation/) on how to install `uv`.
It's quite powerful and speedy package manager written in Rust

### Requirements

```bash
# requirements.txt
pandas
scikit-learn
notebook
seaborn
pyarrow
```

### Set up Python environment

```bash
uv venv --python 3.9.7 # install python 3.9.7 that is used in the course
source .venv/bin/activate # activate the environment
python -V # should be 3.9.7
uv pip install -r requirements.txt
jupyter notebook # run jupyter notebook
```
