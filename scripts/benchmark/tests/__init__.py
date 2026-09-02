"""Makes the test package importable by name.

`test_cli` exercises `--model package.module:attribute` by pointing it at a model
defined in `test_cli` itself, which requires this directory to be a real package. That
is the only reason this file exists.
"""
