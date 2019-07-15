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

# Root directory of edgetpu-native.
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
TAG_PY35 := coral/cross-py3.5
TAG_PY36 := coral/cross-py3.6
PY3_VER ?= $(shell python3 -c "import sys;print('%d%d' % sys.version_info[:2])")

.PHONY: all \
        docker-image-py35 \
        docker-shell-py35 \
        docker-compile-py35 \
        docker-image-py36 \
        docker-shell-py36 \
        docker-compile-py36 \
        clean

all::
	@echo "make docker-image-py35                 - Build docker image for python3.5"
	@echo "make docker-shell-py35                 - Run shell to docker image for python3.5"
	@echo "make docker-image-py36                 - Build docker image for python3.6"
	@echo "make docker-shell-py36                 - Run shell to docker image for python3.6"

init-tf:
	cd $(ROOT_DIR) && git submodule update --init

docker-image-py35: init-tf
	docker build -t $(TAG_PY35) -f $(ROOT_DIR)/tools/Dockerfile.16.04 $(ROOT_DIR)/tools

docker-shell-py35: docker-image-py35
	docker run --rm -it -v $(ROOT_DIR):/edgetpu-native $(TAG_PY35)

docker-image-py36: init-tf
	docker build -t $(TAG_PY36) -f $(ROOT_DIR)/tools/Dockerfile.18.04 $(ROOT_DIR)/tools

docker-shell-py36: docker-image-py36
	docker run --rm -it -v $(ROOT_DIR):/edgetpu-native $(TAG_PY36)

UID ?= $(shell id -u)
GID ?= $(shell id -g)
MKDIR := install -d -m 755 -o $(UID) -g $(GID)
COPY :=  install -C -m 644 -o $(UID) -g $(GID)
BINARY_COPY :=  install -C -m 755 -o $(UID) -g $(GID)
BAZEL_FLAGS := -c opt \
               --verbose_failures \
               --sandbox_debug \
               --crosstool_top=//tools/arm_compiler:toolchain \
               --compiler=clang \
               --linkopt="-Wl,--strip-all" \
               --define PY3_VER=$(PY3_VER)
AMD64_BAZEL_FLAGS := $(BAZEL_FLAGS) \
                     --features=glibc_compat \
                     --cpu=k8
ARM64_BAZEL_FLAGS := $(BAZEL_FLAGS) \
                     --cpu=arm64-v8a
ARM32_BAZEL_FLAGS := $(BAZEL_FLAGS) \
                     --cpu=armeabi-v7a

AMD64_OUT_DIR := bazel-out/k8-opt/bin
ARM64_OUT_DIR := bazel-out/arm64-v8a-opt/bin
ARM32_OUT_DIR := bazel-out/armeabi-v7a-opt/bin

# Debugging util, print variable names. For example, `make print-ROOT_DIR`.
print-%:
	@echo $* = $($*)

define build_for_qa_test =
	@echo "build for QA test: $(1)"
	cd $(ROOT_DIR) && bazel build $(AMD64_BAZEL_FLAGS) $(1)
	$(COPY) $(ROOT_DIR)/$(AMD64_OUT_DIR)/$(1) $(QA_DIR)/x86_64/$(2)
	# "-lm" is required for edgetpu_resource_manager_test.
	cd $(ROOT_DIR) && bazel build $(ARM64_BAZEL_FLAGS) --linkopt="-lm" $(1)
	$(COPY) $(ROOT_DIR)/$(ARM64_OUT_DIR)/$(1) $(QA_DIR)/arm64/$(2)
	cd $(ROOT_DIR) && bazel build $(ARM32_BAZEL_FLAGS) --linkopt="-lm" $(1)
	$(COPY) $(ROOT_DIR)/$(ARM32_OUT_DIR)/$(1) $(QA_DIR)/arm32/$(2)
endef

define binary_cp_dir =
	@echo "Copying directory $(1) to $(2)"
	cp -r $(1) $(2)
	chmod 755 -R $(2)
endef
