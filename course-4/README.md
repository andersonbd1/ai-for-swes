The code in HomeMatch.py follows the steps in the instructions and stores
files inline (named accordingly) with each step so that don't need to redo each step.

Setup the python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the code from scratch, just delete the files and run the code:
```bash
rm -fR ./data/*
python HomeMatch.py
```

You can delete any files in data or none and rerun the code.
```bash
python HomeMatch.py | tail -n 1 | jq -C . | less -R
```
