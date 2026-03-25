PYTHON ?= python

.PHONY: proto clean

proto:
	$(PYTHON) scripts/proto.py generate

clean:
	$(PYTHON) scripts/proto.py clean
