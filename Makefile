# Convenience wrapper. Real build logic lives in CMakeLists.txt / setup.py.
BUILD ?= build
ARCH  ?= 121

.PHONY: all configure build bench python test ncu results clean

all: build

configure:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)

build: configure
	cmake --build $(BUILD) -j

bench: build
	./scripts/run_all_benches.sh $(BUILD)

results:
	python3 scripts/make_results_table.py
	python3 scripts/roofline.py

python:
	pip install -e . --no-build-isolation -v

test: python
	pytest -q tests

ncu: build
	./scripts/profile_ncu.sh $(BUILD)

clean:
	rm -rf $(BUILD) build_ext *.egg-info python/spark_kernels/*.so
