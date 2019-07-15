# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CUR_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
include $(CUR_DIR)/common.mk

MODELS_TOOL_OUT_DIR := $(ROOT_DIR)/models_tool
EXAMPLE_OUT_DIR := $(ROOT_DIR)/cpp_example_out
ALL_BINARY_OUT_DIR := $(ROOT_DIR)/all_binary_out

.PHONY: all \
        docker-build-all \
        docker-join-tflite-models-tool \
        docker-build-examples \
        build-all \
        join-tflite-models-tool \
        build-examples \
        clean

all::
	@echo "make docker-build-all                  - Build everything under edgetpu/cpp for all platforms, must invoke from host"
	@echo "make docker-join-tflite-models-tool    - Build join_tflite_models tool for all platforms, must invoke from host"
	@echo "make docker-build-examples             - Build C++ examples in edgetpu/cpp/examples for all platforms, must invoke from host"
	@echo "make build-all                         - Build everything under edgetpu/cpp for all platforms, must invoke inside Docker shell"
	@echo "make join-tflite-models-tool           - Build join_tflite_models tool for all platforms, must invoke inside Docker shell"
	@echo "make build-examples                    - Build C++ examples in edgetpu/cpp/examples for all platforms, must invoke inside Docker shell"
	@echo "make clean                             - Remove generated files, might need sudo"

docker-build-all: docker-image-py35
	docker run --rm -t -v $(ROOT_DIR):/edgetpu-native $(TAG_PY35) make UID=$(UID) GID=$(GID) -C /edgetpu-native build-all

docker-join-tflite-models-tool: docker-image-py35
	docker run --rm -t -v $(ROOT_DIR):/edgetpu-native $(TAG_PY35) make UID=$(UID) GID=$(GID) -C /edgetpu-native join-tflite-models-tool

docker-build-examples: docker-image-py35
	docker run --rm -t -v $(ROOT_DIR):/edgetpu-native $(TAG_PY35) make UID=$(UID) GID=$(GID) -C /edgetpu-native build-examples

build-all:
	$(MKDIR) $(ALL_BINARY_OUT_DIR)
	# amd64
	$(MKDIR) $(ALL_BINARY_OUT_DIR)/amd64
	bazel build $(AMD64_BAZEL_FLAGS) //edgetpu/cpp/...
	$(call binary_cp_dir,$(AMD64_OUT_DIR)/edgetpu,$(ALL_BINARY_OUT_DIR)/amd64)
	# arm64
	$(MKDIR) $(ALL_BINARY_OUT_DIR)/arm64
	# -lm is required for benchmark library.
	bazel build $(ARM64_BAZEL_FLAGS) --linkopt="-lm" //edgetpu/cpp/...
	$(call binary_cp_dir,$(ARM64_OUT_DIR)/edgetpu,$(ALL_BINARY_OUT_DIR)/arm64)
	# arm32
	$(MKDIR) $(ALL_BINARY_OUT_DIR)/arm32
	# -lm is required for benchmark library.
	bazel build $(ARM32_BAZEL_FLAGS) --linkopt="-lm" //edgetpu/cpp/...
	$(call binary_cp_dir,$(ARM32_OUT_DIR)/edgetpu,$(ALL_BINARY_OUT_DIR)/arm32)

join-tflite-models-tool:
	$(MKDIR) $(MODELS_TOOL_OUT_DIR)
	$(MKDIR) $(MODELS_TOOL_OUT_DIR)/amd64
	bazel build $(AMD64_BAZEL_FLAGS) //edgetpu/cpp/tools:join_tflite_models
	$(BINARY_COPY) $(AMD64_OUT_DIR)/edgetpu/cpp/tools/join_tflite_models $(MODELS_TOOL_OUT_DIR)/amd64/join_tflite_models
	$(MKDIR) $(MODELS_TOOL_OUT_DIR)/arm64
	bazel build $(ARM64_BAZEL_FLAGS) //edgetpu/cpp/tools:join_tflite_models
	$(BINARY_COPY) $(ARM64_OUT_DIR)/edgetpu/cpp/tools/join_tflite_models $(MODELS_TOOL_OUT_DIR)/arm64/join_tflite_models
	$(MKDIR) $(MODELS_TOOL_OUT_DIR)/arm32
	bazel build $(ARM32_BAZEL_FLAGS) //edgetpu/cpp/tools:join_tflite_models
	$(BINARY_COPY) $(ARM32_OUT_DIR)/edgetpu/cpp/tools/join_tflite_models $(MODELS_TOOL_OUT_DIR)/arm32/join_tflite_models

build-examples:
	$(MKDIR) $(EXAMPLE_OUT_DIR)
	# amd64
	$(MKDIR) $(EXAMPLE_OUT_DIR)/x86_64
	bazel build $(AMD64_BAZEL_FLAGS) //edgetpu/cpp/examples:two_models_one_tpu
	$(BINARY_COPY) $(AMD64_OUT_DIR)/edgetpu/cpp/examples/two_models_one_tpu $(EXAMPLE_OUT_DIR)/x86_64/
	bazel build $(AMD64_BAZEL_FLAGS) //edgetpu/cpp/examples:two_models_two_tpus_threaded
	$(BINARY_COPY) $(AMD64_OUT_DIR)/edgetpu/cpp/examples/two_models_two_tpus_threaded $(EXAMPLE_OUT_DIR)/x86_64/
	# arm64
	$(MKDIR) $(EXAMPLE_OUT_DIR)/arm64
	bazel build $(ARM64_BAZEL_FLAGS) //edgetpu/cpp/examples:two_models_one_tpu
	$(BINARY_COPY) $(ARM64_OUT_DIR)/edgetpu/cpp/examples/two_models_one_tpu $(EXAMPLE_OUT_DIR)/arm64/
	bazel build $(ARM64_BAZEL_FLAGS) //edgetpu/cpp/examples:two_models_two_tpus_threaded
	$(BINARY_COPY) $(ARM64_OUT_DIR)/edgetpu/cpp/examples/two_models_two_tpus_threaded $(EXAMPLE_OUT_DIR)/arm64/
	# arm32
	$(MKDIR) $(EXAMPLE_OUT_DIR)/arm32
	bazel build $(ARM32_BAZEL_FLAGS) //edgetpu/cpp/examples:two_models_one_tpu
	$(BINARY_COPY) $(ARM32_OUT_DIR)/edgetpu/cpp/examples/two_models_one_tpu $(EXAMPLE_OUT_DIR)/arm32/
	bazel build $(ARM32_BAZEL_FLAGS) //edgetpu/cpp/examples:two_models_two_tpus_threaded
	$(BINARY_COPY) $(ARM32_OUT_DIR)/edgetpu/cpp/examples/two_models_two_tpus_threaded $(EXAMPLE_OUT_DIR)/arm32/

clean:
	rm -rf $(MODELS_TOOL_OUT_DIR)
	rm -rf $(EXAMPLE_OUT_DIR)
	rm -rf $(ALL_BINARY_OUT_DIR)
