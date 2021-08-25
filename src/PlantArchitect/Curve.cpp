#include <Curve.hpp>

using namespace PlantArchitect;

void Curve::GetUniformCurve(size_t pointAmount, std::vector<glm::vec3> &points) const {
    float step = 1.0f / (pointAmount - 1);
    for (size_t i = 0; i <= pointAmount; i++) {
        points.push_back(GetPoint(step * i));
    }
}


BezierCurve::BezierCurve(glm::vec3 cp0, glm::vec3 cp1, glm::vec3 cp2, glm::vec3 cp3)
        : Curve(),
          m_p0(cp0),
          m_p1(cp1),
          m_p2(cp2),
          m_p3(cp3) {
}

glm::vec3 BezierCurve::GetPoint(float t) const {
    t = glm::clamp(t, 0.f, 1.f);
    return m_p0 * (1.0f - t) * (1.0f - t) * (1.0f - t)
           + m_p1 * 3.0f * t * (1.0f - t) * (1.0f - t)
           + m_p2 * 3.0f * t * t * (1.0f - t)
           + m_p3 * t * t * t;
}

glm::vec3 BezierCurve::GetAxis(float t) const {
    t = glm::clamp(t, 0.f, 1.f);
    float mt = 1.0f - t;
    return (m_p1 - m_p0) * 3.0f * mt * mt + 6.0f * t * mt * (m_p2 - m_p1) + 3.0f * t * t * (m_p3 - m_p2);
}

glm::vec3 BezierCurve::GetStartAxis() const {
    return glm::normalize(m_p1 - m_p0);
}

glm::vec3 BezierCurve::GetEndAxis() const {
    return glm::normalize(m_p3 - m_p2);
}