#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    struct PLANT_ARCHITECT_API InternodeRingSegment {
        glm::vec3 m_startPosition, m_endPosition;
        glm::vec3 m_startAxis, m_endAxis;
        float m_startRadius, m_endRadius;

        InternodeRingSegment(glm::vec3 startPosition, glm::vec3 endPosition,
                             glm::vec3 startAxis, glm::vec3 endAxis,
                             float startRadius, float endRadius);

        void AppendPoints(std::vector<Vertex> &vertices, glm::vec3 &normalDir,
                          int step);

        inline glm::vec3 GetPoint(glm::vec3 &normalDir, float angle, bool isStart);
    };

} // namespace PlantFactory