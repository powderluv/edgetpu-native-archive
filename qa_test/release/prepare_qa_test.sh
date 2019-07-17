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
# This script will compile binaries and prepare test data for C++ QA tests of
# release branch. All files will be stored in qa_test folder.
#
# Steps:
#  1. Run this script under edgetpu-native folder:
#       bash ./qa_test/release/prepare_qa_test.sh
#     Pass IP address and user name via '-d' and '-u' when you want to test on
#     remote machine(Rp3 or Devboard):
#       bash ./qa_test/release/prepare_qa_test.sh -d 192.168.100.2 -u mendel
#
#  2. Tests and data are stored under edgetpu-native/qa_test folder. If you
#     specified IP address and user name, qa_test folder will be copied to HOME
#     directory of the remote machine.
#
#  3. Go to qa_tests and execute run_tests.sh to start the QA test:
#       cd qa_test/release
#       bash ./run_tests.sh
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../"
set -x
set -e
usage() {
  echo "Invalid option!";
  echo "Usage: $0 [-d <IP address of destination device> -u <User name for SCP>]" 1>&2;
  exit 1;
}
# Default IP and user name are for Devboard.
DST_IPADDR=""
USER_NAME=""
while getopts ":d:u:" o; do
  case "${o}" in
    d)
      DST_IPADDR=${OPTARG}
      ;;
    u)
      USER_NAME=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done

make -f ${ROOT_DIR}/qa_test/release/Makefile docker-build-qa-test
make -f ${ROOT_DIR}/Makefile docker-build-examples
cp -r ${ROOT_DIR}/cpp_example_out/* ${ROOT_DIR}/qa_test

mkdir -p ${ROOT_DIR}/qa_test/cpp_example_data

# Copy test data.
cp -R ${ROOT_DIR}/edgetpu/cpp/basic/test_data  ${ROOT_DIR}/qa_test
cp -R ${ROOT_DIR}/edgetpu/cpp/basic/invalid_models  ${ROOT_DIR}/qa_test
cp -R ${ROOT_DIR}/edgetpu/cpp/learn/imprinting/test_data ${ROOT_DIR}/qa_test/imprinting_test_data
cp -R ${ROOT_DIR}/edgetpu/cpp/posenet/test_data ${ROOT_DIR}/qa_test/posenet_test_data

# Copy models for cpp example.
cp ${ROOT_DIR}/edgetpu/cpp/basic/test_data/bird.bmp ${ROOT_DIR}/qa_test/cpp_example_data/bird.bmp
cp ${ROOT_DIR}/edgetpu/cpp/basic/test_data/sunflower.bmp ${ROOT_DIR}/qa_test/cpp_example_data/plant.bmp
cp ${ROOT_DIR}/edgetpu/cpp/basic/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite ${ROOT_DIR}/qa_test/cpp_example_data/inat_bird_edgetpu.tflite
cp ${ROOT_DIR}/edgetpu/cpp/basic/test_data/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite ${ROOT_DIR}/qa_test/cpp_example_data/inat_plant_edgetpu.tflite

# Copy runtime.
cp -R ${ROOT_DIR}/libedgetpu ${ROOT_DIR}/qa_test
# Set permissions.
chmod -R 755 ${ROOT_DIR}/qa_test/x86_64/*
chmod -R 755 ${ROOT_DIR}/qa_test/arm64/*
chmod -R 755 ${ROOT_DIR}/qa_test/arm32/*

if [[ $DST_IPADDR ]] && [[ $USER_NAME ]] ; then
  echo "We'll copy QA tests to : $DST_IPADDR ."
  scp -r ${ROOT_DIR}/qa_test ${USER_NAME}@${DST_IPADDR}:~
fi
echo "Release branch QA tests successfully prepared."
