#!/bin/bash

if ! conda activate /home/featurize/work/mch/env 2>/dev/null; then
    echo "Run with 'source ./configure.sh'" >&2
    exit 1
fi

pip install --user -r requirements.txt
pip install --user -r requirements_demo.txt
pip install --user "httpx[socks]"

pip install --user ninja
MAX_JOBS=4 pip install --user --no-build-isolation --verbose flash-attn

featurize port export 7860
