"""Detector wrappers evaluated by the offline benchmark.

Each module here exposes one callable matching the harness contract in `cli.py`: it
takes a `dataset.Clip` and returns a float where higher means *more likely
manipulated*. Nothing registers itself and nothing inherits from anything — `--model`
names the callable by its dotted path, and that is the whole mechanism.
"""
