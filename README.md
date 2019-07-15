# edgetpu-native

This repo contains code for Edge TPU support for Coral product. See
[coral.withgoogle.com](https://coral.withgoogle.com) for more details.

## Contents

This repo contains the c++ source for Edge TPU. In particular, it includes a
cross-compile tool configuration (through bazel inside docker image) to build
targets for different platforms, including amd64, arm64, arm32.

`Makefile` contains some convenient commands to build a group of binaries. For
example,

*   `make docker-build-examples` will build all examples under
    edgetpu/cpp/examples folder.

*   `make docker-build-all` will build everything under edgetpu/cpp folder

*   `make docker-shell-py35` will open a docker shell and mount current folder
    under `/edgetpu-native`

Note: rules that start with `docker` should be invoked from host machine.

Additionally, you can build all of the tests and benchmarks with
`./qa_test/prepare_qa_test.sh` and run all of them with `./qa_test/run_tests.sh`

For details about C++ API, checkout our
[docs](https://coral.withgoogle.com/docs/edgetpu/api-cpp/)
