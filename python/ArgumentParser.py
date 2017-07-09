#!/usr/bin/python

import sys
from argparse import ArgumentParser

p = ArgumentParser(usage='it is usage tip', description='this is a test')
p.add_argument('--one', default=1, type=int, help='the first argument')
p.add_argument('--two', default=2, type=int, help='the second argument')
p.add_argument('--docs-dir', default="./", help='document directory')

args, remaining = p.parse_known_args(sys.argv)

print args
print remaining

