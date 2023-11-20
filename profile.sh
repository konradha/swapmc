#!/bin/bash
python -m cProfile -o data.prof gblas.py
flameprof data.prof > profile.svg
