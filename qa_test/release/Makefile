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
include $(CUR_DIR)/../../common.mk

QA_DIR := $(ROOT_DIR)/qa_test

.PHONY: all \
	docker-build-qa-test \
	build-qa-test

all::
	@echo "make docker-build-qa-test     - Build QA tests for all platforms in Docker"
	@echo "make build-qa-test            - Build tests and benchmarks for QA"

docker-build-qa-test: docker-image-py35
	docker run --rm -t -v $(ROOT_DIR):/edgetpu-native $(TAG_PY35) make UID=$(UID) GID=$(GID) -C /edgetpu-native/qa_test/release build-qa-test

build-qa-test:
	$(MKDIR) $(QA_DIR)/x86_64
	$(MKDIR) $(QA_DIR)/arm32
	$(MKDIR) $(QA_DIR)/arm64
	$(call build_for_qa_test,edgetpu/cpp/error_reporter_test)
	$(call build_for_qa_test,edgetpu/cpp/version_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/basic_engine_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/basic_engine_native_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/edgetpu_resource_manager_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/edgetpu_resource_manager_benchmark)
	$(call build_for_qa_test,edgetpu/cpp/basic/models_benchmark)
	$(call build_for_qa_test,edgetpu/cpp/basic/models_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/inference_repeatability_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/model_loading_stress_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/inference_stress_test)
	$(call build_for_qa_test,edgetpu/cpp/classification/engine_test,classification_engine_test)
	$(call build_for_qa_test,edgetpu/cpp/classification/models_test,classification_models_test)
	$(call build_for_qa_test,edgetpu/cpp/detection/engine_test,detection_engine_test)
	$(call build_for_qa_test,edgetpu/cpp/detection/models_test,detection_models_test)
	$(call build_for_qa_test,edgetpu/cpp/learn/imprinting/engine_test,imprinting_engine_test)
	$(call build_for_qa_test,edgetpu/cpp/learn/utils_test)
	$(call build_for_qa_test,edgetpu/cpp/posenet/models_benchmark,posenet_benchmark)
	$(call build_for_qa_test,edgetpu/cpp/posenet/models_test,posenet_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/multiple_tpus_inference_stress_test)
	$(call build_for_qa_test,edgetpu/cpp/basic/multiple_tpus_performance_analysis)
