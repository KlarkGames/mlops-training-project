prepare_stage1:
	uv run python src/data/data_download.py

prepare_stage2:
	uv run python src/data/data_prepare.py

data_prepare: prepare_stage1 prepare_stage2

training:
	uv run python src/models/model_train.py
