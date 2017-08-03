#!/usr/bin/env bash
THEANO_FLAGS='floatX=float32,device=cuda2,optimizer=fast_run' python main.py --cfgfile ./configurations/model.json
