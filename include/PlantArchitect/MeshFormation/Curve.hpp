#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class PLANT_ARCHITECT_API Curve {
    public:
        virtual glm::vec3 GetPoint(float t) const = 0;

        virtual glm::vec3 GetAxis(float t) const = 0;

        void GetUniformCurve(size_t pointAmount,
                             std::vector<glm::vec3> &points) const;
    };

    class PLANT_ARCHITECT_API BezierCurve : public Curve {
    public:
        BezierCurve() = default;
        BezierCurve(glm::vec3 cp0, glm::vec3 cp1, glm::vec3 cp2, glm::vec3 cp3);

        [[nodiscard]] glm::vec3 GetPoint(float t) const override;

        [[nodiscard]] glm::vec3 GetAxis(float t) const override;

        [[nodiscard]] glm::vec3 GetStartAxis() const;

        [[nodiscard]] glm::vec3 GetEndAxis() const;

        glm::vec3 m_p0, m_p1, m_p2, m_p3;
    };
} // namespace PlantFactory
