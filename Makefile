.PHONY: setup lint test

VENV?=.venv
PY=python3
PIP=$(VENV)/bin/pip
PYBIN=$(VENV)/bin/python
CONDA?=0
CONDA_PREFIX?=

setup:
ifeq ($(CONDA),1)
	@test -n "$(CONDA_PREFIX)" || (echo "CONDA=1 but CONDA_PREFIX is empty. Activate your conda env first." && false)
	@echo "Using conda env at $(CONDA_PREFIX)"
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
else
	$(MAKE) $(VENV)/bin/activate
endif

$(VENV)/bin/activate: requirements.txt
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint: $(VENV)/bin/activate
	$(PYBIN) -m pip install ruff
	$(VENV)/bin/ruff check src tests

test:
ifeq ($(CONDA),1)
	python -m pip install pytest
	PYTHONPATH=src python -m pytest -q
else
	$(PYBIN) -m pip install pytest
	PYTHONPATH=src $(VENV)/bin/pytest -q
endif

report:
	PYTHONPATH=src python scripts/build_report.py
