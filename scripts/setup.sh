#!/bin/bash
# Download spaCy language model
pip install -r scripts/requirements.txt
python -m spacy download en_core_web_sm
