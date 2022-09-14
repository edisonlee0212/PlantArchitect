#pragma once

#include "Application.hpp"
#include "plant_architect_export.h"
#include "Curve.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    class StrandSegment;
    class PLANT_ARCHITECT_API StrandKnot {
        std::weak_ptr<StrandKnot> m_prev;
        std::weak_ptr<StrandKnot> m_next;
    public:
        glm::vec3 m_position = glm::vec3(0.0f);
    };

    class PLANT_ARCHITECT_API Strand {
        friend class StrandPlant;
        std::weak_ptr<StrandKnot> m_start;
        std::weak_ptr<StrandKnot> m_end;
    public:

    };

    class PLANT_ARCHITECT_API StrandsIntersection {
        friend class StrandPlant;
        std::vector<std::shared_ptr<StrandKnot>> m_knots;
        std::vector<std::shared_ptr<StrandsIntersection>> m_children;
    public:

    };

    struct PLANT_ARCHITECT_API StrandsIntersectionRegion{
        std::vector<glm::vec2> m_regionPoints;
        glm::vec2 m_radius = glm::vec2(0.0f);
        StrandsIntersectionRegion();
        void Construct(const std::vector<glm::vec2>& points);
        [[nodiscard]] bool IsInRegion(const glm::vec2& point) const;

    };

    class PLANT_ARCHITECT_API StrandPlant : public IPrivateComponent{
        std::vector<std::shared_ptr<Strand>> m_strands;
        std::shared_ptr<StrandsIntersection> m_root;
        StrandsIntersectionRegion m_rootRegion;
        float m_pointDistance = 0.01f;
    public:
        void OnInspect() override;
        void GenerateStrands(float pointDistance = 0.01f);
    };
}