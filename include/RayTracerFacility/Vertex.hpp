#pragma once
#include <glm/glm.hpp>
namespace RayTracerFacility {
struct Vertex {
  glm::vec3 m_position;
  glm::vec3 m_normal;
  glm::vec3 m_tangent;
  glm::vec4 m_color;
  glm::vec2 m_texCoords;
};
} // namespace RayTracerFacility