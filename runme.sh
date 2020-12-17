#!/bin/bash

#python3 -m venv my_venv
#source my_venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=my_venv
jupyter notebook --notebook-dir