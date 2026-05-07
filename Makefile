SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

-include .env

APP_NAME ?= mnist_sensehat_demo
ARTIFACT_DIR ?= artifacts
BUILD_DIR ?= build
BUILD_JOBS ?= $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
TFLITE_SRC_DIR ?= third_party/tensorflow-src
TF_TAG ?= v2.16.1
PYTHON_BIN ?= .venv/bin/python
PIP_BIN ?= .venv/bin/pip
TRAIN_ARGS ?=
VENV_STAMP := .venv/.deps-stamp

MAKEFLAGS += -j$(BUILD_JOBS)
export CMAKE_BUILD_PARALLEL_LEVEL := $(BUILD_JOBS)

APP_BINARY := $(BUILD_DIR)/$(APP_NAME)
MODEL_FILE := $(ARTIFACT_DIR)/model.tflite
TFLITE_READY := $(TFLITE_SRC_DIR)/.source-ready
ARTIFACT_DIR_STAMP := $(ARTIFACT_DIR)/.dir-stamp
BUILD_DIR_STAMP := $(BUILD_DIR)/.dir-stamp
CPP_SOURCES := $(shell find src -type f \( -name '*.cpp' -o -name '*.h' \) 2>/dev/null)

.PHONY: all train tflite build deploy run perf perf-report provision-pi console clean

all: deploy

$(ARTIFACT_DIR_STAMP):
	mkdir -p "$(ARTIFACT_DIR)"
	touch "$@"

$(BUILD_DIR_STAMP):
	mkdir -p "$(BUILD_DIR)"
	touch "$@"

$(VENV_STAMP): requirements.txt
	if [[ ! -x "$(PYTHON_BIN)" ]]; then python3 -m venv .venv; fi
	"$(PYTHON_BIN)" -m pip install --upgrade pip setuptools wheel
	"$(PIP_BIN)" install -r requirements.txt
	touch "$@"

$(MODEL_FILE): scripts/train_and_convert.py requirements.txt $(VENV_STAMP) | $(ARTIFACT_DIR_STAMP)
	"$(PYTHON_BIN)" scripts/train_and_convert.py --artifacts-dir "$(ARTIFACT_DIR)" $(TRAIN_ARGS)

train: $(MODEL_FILE)

$(TFLITE_READY): scripts/ensure_tflite_source.sh .env
	bash scripts/ensure_tflite_source.sh

tflite: $(TFLITE_READY)

$(APP_BINARY): CMakeLists.txt toolchains/aarch64.cmake $(CPP_SOURCES) $(TFLITE_READY) | $(BUILD_DIR_STAMP)
	cmake -S . -B "$(BUILD_DIR)" -G Ninja \
	  -DCMAKE_TOOLCHAIN_FILE="$(abspath toolchains/aarch64.cmake)" \
	  -DCMAKE_BUILD_TYPE=Debug \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	  -DAPP_NAME="$(APP_NAME)" \
	  -DTFLITE_SRC_DIR="$(abspath $(TFLITE_SRC_DIR))"
	cmake --build "$(BUILD_DIR)" --parallel "$(BUILD_JOBS)" --target "$(APP_NAME)"

build: $(APP_BINARY)

deploy: $(APP_BINARY) $(MODEL_FILE)
	bash scripts/deploy_to_pi.sh

run: $(APP_BINARY) $(MODEL_FILE)
	bash scripts/run_on_pi.sh

perf: $(APP_BINARY) $(MODEL_FILE)
	bash scripts/run_perf_on_pi.sh

perf-report:
	"$(PYTHON_BIN)" scripts/parse_perf_results.py --input-dir "$(ARTIFACT_DIR)/perf" --model-metrics "$(ARTIFACT_DIR)/model_metrics.csv" --output "$(ARTIFACT_DIR)/perf/results_table.csv"

provision-pi:
	bash scripts/provision_pi.sh

console:
	bash scripts/open_pi_console.sh

clean:
	rm -rf "$(BUILD_DIR)" "$(ARTIFACT_DIR)" "third_party/tensorflow-src" "third_party/litert-build" "third_party/litert"
