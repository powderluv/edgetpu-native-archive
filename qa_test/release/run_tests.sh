#!/bin/bash
#
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
#
set -e
set -x
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CLEAR="\033[0m"
function info {
  echo -e "${GREEN}${1}${CLEAR}"
}
function warn {
  echo -e "${YELLOW}${1}${CLEAR}"
}
function error {
  echo -e "${RED}${1}${CLEAR}"
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../"
cpu_arch=$(uname -m)
os_version=$(uname -v)
if [[ "$cpu_arch" == "x86_64" ]] && [[ "$os_version" == *"Debian"* || "$os_version" == *"Ubuntu"* ]]; then
  platform="x86_64"
  LIBEDGETPU_SUFFIX=x86_64
  HOST_GNU_TYPE=x86_64-linux-gnu
  info "Recognized as Linux on x86_64!"
elif [[ "$cpu_arch" == "armv7l" ]]; then
  board_version=$(cat /proc/device-tree/model)
  if [[ "$board_version" == "Raspberry Pi 3 Model B Rev"* ]]; then
    platform="arm32"
    LIBEDGETPU_SUFFIX=arm32
    HOST_GNU_TYPE=arm-linux-gnueabihf
    info "Recognized as Raspberry Pi 3 B!"
  elif [[ "$board_version" == "Raspberry Pi 3 Model B Plus Rev"* ]]; then
    platform="arm32"
    LIBEDGETPU_SUFFIX=arm32
    HOST_GNU_TYPE=arm-linux-gnueabihf
    info "Recognized as Raspberry Pi 3 B+!"
  fi
elif [[ -f /etc/mendel_version ]]; then
  platform="arm64"
  LIBEDGETPU_SUFFIX=arm64
  HOST_GNU_TYPE=aarch64-linux-gnu
  info "Recognized as Edge TPU DevBoard!"
else
  error "Platform not supported!"
  exit 1
fi
LIBEDGETPU_SRC="${SCRIPT_DIR}/../libedgetpu/libedgetpu_${LIBEDGETPU_SUFFIX}.so"
LIBEDGETPU_DST="/usr/lib/${HOST_GNU_TYPE}/libedgetpu.so.1.0"
# Runtime library.
info "Installing Edge TPU runtime library [${LIBEDGETPU_DST}]..."
if [[ -f "${LIBEDGETPU_DST}" ]]; then
  warn "File already exists. Replacing it..."
  sudo rm -f "${LIBEDGETPU_DST}"
fi
sudo cp -p "${LIBEDGETPU_SRC}" "${LIBEDGETPU_DST}"
sudo ldconfig
info "Done"
function run_tests() {
  "${ROOT_DIR}/qa_test/${platform}"/models_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/models_benchmark \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --benchmark_out="${ROOT_DIR}/qa_test/benchmark_${platform}.csv" \
    --benchmark_out_format=csv
  "${ROOT_DIR}/qa_test/${platform}"/inference_repeatability_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --stress_test_runs=20
  "${ROOT_DIR}/qa_test/${platform}"/model_loading_stress_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --stress_test_runs=20
  "${ROOT_DIR}/qa_test/${platform}"/inference_stress_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --stress_test_runs=20 \
    --stress_with_sleep_test_runs=5
  "${ROOT_DIR}/qa_test/${platform}"/error_reporter_test
  "${ROOT_DIR}/qa_test/${platform}"/basic_engine_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/edgetpu_resource_manager_test
  "${ROOT_DIR}/qa_test/${platform}"/edgetpu_resource_manager_benchmark
  "${ROOT_DIR}/qa_test/${platform}"/version_test
  "${ROOT_DIR}/qa_test/${platform}"/basic_engine_native_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/classification_engine_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/classification_models_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/detection_engine_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/detection_models_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/imprinting_engine_test \
    --imprinting_data_dir="${ROOT_DIR}/qa_test/imprinting_test_data"
  "${ROOT_DIR}/qa_test/${platform}"/utils_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
  "${ROOT_DIR}/qa_test/${platform}"/posenet_benchmark \
    --posenet_model_dir="${ROOT_DIR}/qa_test/posenet_test_data"
  "${ROOT_DIR}/qa_test/${platform}"/posenet_test \
    --posenet_model_dir="${ROOT_DIR}/qa_test/posenet_test_data" \
    --posenet_data_dir="${ROOT_DIR}/qa_test/posenet_test_data"
}
run_tests

info "To run the following tests, please insert additional Edge TPU if only one Edge TPU is connected right now."
info "Press Enter when ready."
read LINE
function multiple_edgetpu_tests() {
  "${ROOT_DIR}/qa_test/${platform}"/multiple_tpus_inference_stress_test \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"

  "${ROOT_DIR}/qa_test/${platform}"/multiple_tpus_performance_analysis \
    --model_dir="${ROOT_DIR}/qa_test/test_data" \
    --test_data_dir="${ROOT_DIR}/qa_test/test_data"
}
multiple_edgetpu_tests


function cpp_example_tests() {
  cd /tmp
  if [[ -d edgetpu_cpp_example ]]; then
    rm -Rf edgetpu_cpp_example
  fi
  mkdir edgetpu_cpp_example
  cp ${ROOT_DIR}/qa_test/cpp_example_data/* edgetpu_cpp_example
  "${ROOT_DIR}/qa_test/${platform}"/two_models_one_tpu
  "${ROOT_DIR}/qa_test/${platform}"/two_models_two_tpus_threaded
}
cpp_example_tests
