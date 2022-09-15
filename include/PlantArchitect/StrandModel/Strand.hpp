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
        std::weak_ptr<StrandKnot> m_upLeft;
        std::weak_ptr<StrandKnot> m_upRight;
        std::weak_ptr<StrandKnot> m_right;
        std::weak_ptr<StrandKnot> m_downRight;
        std::weak_ptr<StrandKnot> m_downLeft;
        std::weak_ptr<StrandKnot> m_left;

        int m_distanceToBoundary = 0;
        glm::ivec2 m_coordinate = glm::ivec2(0);
        glm::vec3 m_position = glm::vec3(0.0f);
    };

    class PLANT_ARCHITECT_API Strand {
        friend class StrandPlant;
        friend class StrandsIntersection;
        std::weak_ptr<StrandKnot> m_start;
        std::weak_ptr<StrandKnot> m_end;
    public:

    };

    class PLANT_ARCHITECT_API StrandsIntersection : public IPrivateComponent{
        friend class StrandPlant;
        std::vector<std::shared_ptr<StrandKnot>> m_strandKnots;

        bool m_isRoot = false;
        float m_pointDistance = 0.01f;
        std::vector<glm::vec2> m_regionBoundary;
        glm::vec2 m_boundaryRadius = glm::vec2(0.0f);
        int m_maxDistanceToEnd = 0;
        void DisplayIntersection(const std::string& title, bool editable);
    public:
        void SetPointDistance(float value);

        void FillPoints();
        void CalculateConnectivity();
        void OnCreate() override;
        void OnInspect() override;
        void Construct(const std::vector<glm::vec2>& points);
        [[nodiscard]] bool IsInRegion(const glm::vec2& point) const;
    };

    class PLANT_ARCHITECT_API StrandPlant : public IPrivateComponent{
        std::vector<std::shared_ptr<Strand>> m_strands;
    public:
        [[nodiscard]] Entity GetRoot();
        void OnCreate() override;
        void OnInspect() override;
        void GenerateStrands(float pointDistance = 0.01f);
    };




}