#include <LeafSegment.hpp>

using namespace PlantArchitect;

LeafSegment::LeafSegment(glm::vec3 position, glm::vec3 up, glm::vec3 front, float leafHalfWidth,
                         float theta, bool isLeaf, float leftFlatness, float rightFlatness,
                         float leftFlatnessFactor,
                         float rightFlatnessFactor) {
    m_isLeaf = isLeaf;
    m_position = position;
    m_up = up;
    m_front = front;
    m_leafHalfWidth = leafHalfWidth;
    m_theta = theta;
    m_leftFlatness = leftFlatness;
    m_rightFlatness = rightFlatness;
    m_leftFlatnessFactor = leftFlatnessFactor;
    m_rightFlatnessFactor = rightFlatnessFactor;
    m_radius = theta < 90.0f ? m_leafHalfWidth / glm::sin(glm::radians(m_theta)) : m_leafHalfWidth;
}

glm::vec3 LeafSegment::GetPoint(float angle) {
    if (m_theta < 90.0f) {
        const auto midRibMaxHeight = m_leafHalfWidth / 8.0f;
        const auto midRibMaxAngle = m_theta / 5.0f;

        auto midRibHeight = 0.0f;
        if (glm::abs(angle) < midRibMaxAngle) {
            midRibHeight = midRibMaxHeight * glm::cos(glm::radians(90.0f * glm::abs(angle) / midRibMaxAngle));
        }

        const auto distanceToCenter = m_radius * glm::cos(glm::radians(glm::abs(angle)));
        const auto actualHeight = (m_radius - distanceToCenter) * (angle < 0 ? m_leftFlatness : m_rightFlatness);
        const auto maxHeight = m_radius * (1.0f - glm::cos(glm::radians(glm::abs(m_theta))));
        const auto center = m_position + (m_radius - m_leafHalfWidth) * m_up;
        const auto direction = glm::rotate(m_up, glm::radians(angle), m_front);
        float compressFactor = glm::pow(actualHeight / maxHeight,
                                        angle < 0 ? m_leftFlatnessFactor : m_rightFlatnessFactor);
        if (glm::isnan(compressFactor)) {
            compressFactor = 0.0f;
        }
        return center - m_radius * direction - actualHeight * compressFactor * m_up/* + midRibHeight * Up*/;
    }
    const auto direction = glm::rotate(m_up, glm::radians(angle), m_front);
    return m_position - m_radius * direction;
}
