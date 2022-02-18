# -*- coding: utf-8 -*-
import re
import sys
import argparse
from pylint import lint

THRESHOLD = 8.5

if len(sys.argv) < 2:
    raise argparse.ArgumentError("Module to evaluate needs to be the first argument")

sys.argv[1] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[1])
run = lint.Run([sys.argv[1]], do_exit=False)
score = run.linter.stats['global_note']

if score < THRESHOLD:
    print("Your code doesn't pass the PEP8 style score threshold: %f!" % THRESHOLD)
    sys.exit(1)

print("Congratulations! Your code has passed the PEP8 style score threshold: %f!" % THRESHOLD)