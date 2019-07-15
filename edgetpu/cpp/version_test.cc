#include "edgetpu/cpp/version.h"

#include "gtest/gtest.h"

namespace coral {
TEST(VersionTest, CheckDriverVersion) {
  EXPECT_EQ(GetRuntimeVersion(), kSupportedRuntimeVersion);
}
}  // namespace coral
