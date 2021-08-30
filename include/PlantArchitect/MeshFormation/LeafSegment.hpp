#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API LeafSegment {
    public:
        glm::vec3 m_position;
        glm::vec3 m_front;
        glm::vec3 m_up;
        glm::quat m_rotation;
        float m_leafHalfWidth;
        float m_theta;
        float m_radius;
        float m_leftFlatness;
        float m_leftFlatnessFactor;
        float m_rightFlatness;
        float m_rightFlatnessFactor;
        bool m_isLeaf;

        LeafSegment(glm::vec3 position, glm::vec3 up, glm::vec3 front,
                    float leafHalfWidth, float theta, bool isLeaf,
                    float leftFlatness = 0.0f, float rightFlatness = 0.0f,
                    float leftFlatnessFactor = 1.0f,
                    float rightFlatnessFactor = 1.0f);

        glm::vec3 GetPoint(float angle);
    };
} // namespace PlantFactory