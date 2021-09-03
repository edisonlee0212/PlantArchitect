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
    struct SkinnedVertex {
        glm::vec3 m_position;
        glm::vec3 m_normal;
        glm::vec3 m_tangent;
        glm::vec4 m_color;
        glm::vec2 m_texCoords;

        glm::ivec4 m_bondId;
        glm::vec4 m_weight;
        glm::ivec4 m_bondId2;
        glm::vec4 m_weight2;
    };
} // namespace RayTracerFacility