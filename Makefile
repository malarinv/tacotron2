.PHONY: clean clean-test clean-pyc clean-build docs help common.mk
.DEFAULT_GOAL := help


notebook:
	jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token='${JUPYTER_TOKEN}'
