workspace(name = "edgetpu")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# Be consistent with tensorflow/WORKSPACE.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

# Be consistent with tensorflow/WORKSPACE.
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases

http_archive(
    name = "com_google_glog",
    # For security purpose, can use `sha256sum` on linux to calculate.
    sha256 = "835888ec47ee8065b3098f3ec4373717d641954970f009833ed6d466c397409a",
    strip_prefix = "glog-41f4bf9cbc3e8995d628b459f6a239df43c2b84a",
    urls = [
        "https://github.com/google/glog/archive/41f4bf9cbc3e8995d628b459f6a239df43c2b84a.tar.gz",
    ],
)

# Make sure it matches the definition in glog/WORKSPACE
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.2.tar.gz",
        "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    ],
)

http_archive(
  name = "com_google_absl",
  sha256 = "d10f684f170eb36f3ce752d2819a0be8cc703b429247d7d662ba5b4b48dd7f65",
  strip_prefix = "abseil-cpp-3088e76c597e068479e82508b1770a7ad0c806b6",
  urls = [
    "https://github.com/abseil/abseil-cpp/archive/3088e76c597e068479e82508b1770a7ad0c806b6.tar.gz",
  ],
)

http_archive(
  name = "com_github_google_benchmark",
  sha256 = "59f918c8ccd4d74b6ac43484467b500f1d64b40cc1010daa055375b322a43ba3",
  strip_prefix = "benchmark-16703ff83c1ae6d53e5155df3bb3ab0bc96083be",
  urls = [
    "https://github.com/google/benchmark/archive/16703ff83c1ae6d53e5155df3bb3ab0bc96083be.zip"
  ],
)

local_repository(
    name = "org_tensorflow",
    path = "tensorflow",
)
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace()

local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
)

local_repository(
    name = "edgetpu_swig",
    path = "edgetpu/swig",
)

new_local_repository(
    name = "python_linux",
    path = "/",
    build_file = "BUILD.python"
)
