# For relative imports to work in Python 3.6
def __enable_relative_imports():
	import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
__enable_relative_imports()