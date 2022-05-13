#include <InternodeRingSegment.hpp>

using namespace PlantArchitect;

InternodeRingSegment::InternodeRingSegment(glm::vec3 startPosition, glm::vec3 endPosition, glm::vec3 startAxis,
                                           glm::vec3 endAxis, float startRadius, float endRadius)
        : m_startPosition(startPosition),
          m_endPosition(endPosition),
          m_startAxis(startAxis),
          m_endAxis(endAxis),
          m_startRadius(startRadius),
          m_endRadius(endRadius) {
}

void InternodeRingSegment::AppendPoints(std::vector<Vertex> &vertices, glm::vec3 &normalDir, int step) {
    std::vector<Vertex> startRing;
    std::vector<Vertex> endRing;

    float angleStep = 360.0f / (float) (step);
    Vertex archetype;
    for (int i = 0; i < step; i++) {
        archetype.m_position = GetPoint(normalDir, angleStep * i, true);
        startRing.push_back(archetype);
    }
    for (int i = 0; i < step; i++) {
        archetype.m_position = GetPoint(normalDir, angleStep * i, false);
        endRing.push_back(archetype);
    }
    float textureXstep = 1.0f / step * 4;
    for (int i = 0; i < step - 1; i++) {
        float x = (i % step) * textureXstep;
        startRing[i].m_texCoords = glm::vec2(x, 0.0f);
        startRing[i + 1].m_texCoords = glm::vec2(x + textureXstep, 0.0f);
        endRing[i].m_texCoords = glm::vec2(x, 1.0f);
        endRing[i + 1].m_texCoords = glm::vec2(x + textureXstep, 1.0f);
        vertices.push_back(startRing[i]);
        vertices.push_back(startRing[i + 1]);
        vertices.push_back(endRing[i]);
        vertices.push_back(endRing[i + 1]);
        vertices.push_back(endRing[i]);
        vertices.push_back(startRing[i + 1]);
    }
    startRing[step - 1].m_texCoords = glm::vec2(1.0f - textureXstep, 0.0f);
    startRing[0].m_texCoords = glm::vec2(1.0f, 0.0f);
    endRing[step - 1].m_texCoords = glm::vec2(1.0f - textureXstep, 1.0f);
    endRing[0].m_texCoords = glm::vec2(1.0f, 1.0f);
    vertices.push_back(startRing[step - 1]);
    vertices.push_back(startRing[0]);
    vertices.push_back(endRing[step - 1]);
    vertices.push_back(endRing[0]);
    vertices.push_back(endRing[step - 1]);
    vertices.push_back(startRing[0]);
}

glm::vec3 InternodeRingSegment::GetPoint(glm::vec3 &normalDir, float angle, bool isStart) {
    glm::vec3 direction = glm::cross(normalDir, isStart ? this->m_startAxis : this->m_endAxis);
    direction = glm::rotate(direction, glm::radians(angle), isStart ? this->m_startAxis : this->m_endAxis);
    direction = glm::normalize(direction);
    const glm::vec3 position = (isStart ? m_startPosition : m_endPosition) + direction * (
            isStart ? m_startRadius : m_endRadius);
    return position;
}

InternodeRingSegment::InternodeRingSegment() {

}
