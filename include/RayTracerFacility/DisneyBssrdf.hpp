#pragma once
namespace RayTracerFacility {
struct DisneyBssrdf {
  glm::vec3 R;
  glm::vec3 d;
  DisneyBssrdf(const glm::vec3 &R, const glm::vec3 &mfp) : R(R) {
    // Approximate Reflectance Profiles for Efficient Subsurface Scattering, Eq
    // 6
    const auto s = glm::vec3(1.9f) - R +
                   3.5f * (R - glm::vec3(0.8f)) * (R - glm::vec3(0.8f));
    // prevent the scatter distance to be zero, not a perfect solution, but it
    // works. the divide four pi thing is just to get similar result with Cycles
    // SSS implementation with same inputs.
    const auto l = mfp * 0.07957747f;
    d = glm::clamp(l, 0.0001f, FLT_MAX) / s;
  }
};
} // namespace RayTracerFacility
