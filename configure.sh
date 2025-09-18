#!/bin/bash

conda activate /home/featurize/work/mch/env
pip install --user -r requirements.txt
pip install --user -r requirements_demo.txt
pip install --user "httpx[socks]"

pip install --user ninja
pip install --user --no-build-isolation --verbose flash-attn

featurize port export 7860
