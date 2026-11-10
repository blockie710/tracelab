.PHONY: help install test demo vision verify clean

help:
	@echo "tracelab"
	@echo "  make install   editable install (no runtime deps)"
	@echo "  make test      14 stdlib tests"
	@echo "  make demo      benchmark + HTML dashboard in demo/"
	@echo "  make vision    pixel->degrees round-trip check"
	@echo "  make verify    all of the above, fail on any error"

install:
	python3 -m pip install -e .

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m tracelab.cli demo --out demo/index.html

vision:
	PYTHONPATH=src python3 -m tracelab.cli vision

verify: test demo vision
	@PYTHONPATH=src python3 -c "import json,sys; r=json.load(open('demo/index.json')); \
	 auc=r['metrics']['auc']; print('AUC=%.4f recall@1%%FPR=%.3f' % (auc, r['metrics']['tpr_at_1pct_fpr'])); \
	 sys.exit(0 if auc > 0.95 else 1)"
	@echo "verify: OK"

clean:
	rm -rf demo/index.json demo/frames **/__pycache__ __pycache__
