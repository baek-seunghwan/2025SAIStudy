.PHONY: train infer search submit all

train:
	python -m src.run_train --config config.yaml

infer:
	python -m src.run_infer --config config.yaml

search:
	python -m src.threshold_search --config config.yaml

submit: infer

all: train search infer
