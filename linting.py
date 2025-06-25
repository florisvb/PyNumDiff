# -*- coding: utf-8 -*-
import re, sys, argparse
from pylint import lint

# Call `echo $?` to see the exit code after a run. If the score is over this, the exit
# code will be 0 (success), and if not will be nonzero (failure).
THRESHOLD = 8.5

if len(sys.argv) < 2:
    raise argparse.ArgumentError("Module to evaluate needs to be the first argument")

sys.argv[1] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[1])
run = lint.Run([sys.argv[1], f"--fail-under={THRESHOLD}"])
