#pragma once

#include "Application.hpp"
#include "plant_architect_export.h"
#include "Curve.hpp"

using namespace UniEngine;
namespace PlantArchitect {
    class Strand;
    class PLANT_ARCHITECT_API StrandKnot {
    public:
        std::weak_ptr<Strand> m_strand;

        std::weak_ptr<StrandKnot> m_prev;
        std::weak_ptr<StrandKnot> m_next;

        std::weak_ptr<StrandKnot> m_upLeft;
        std::weak_ptr<StrandKnot> m_upRight;
        std::weak_ptr<StrandKnot> m_right;
        std::weak_ptr<StrandKnot> m_downRight;
        std::weak_ptr<StrandKnot> m_downLeft;
        std::weak_ptr<StrandKnot> m_left;

        bool m_selected = false;
        int m_distanceToBoundary = 0;
        glm::ivec2 m_coordinate = glm::ivec2(0);
    };

    class PLANT_ARCHITECT_API Strand {
    public:
        std::weak_ptr<StrandKnot> m_start;
        std::weak_ptr<StrandKnot> m_end;
    };

    struct SplitSettings{
        int m_knotSize;
        glm::vec2 m_direction;
    };

    class PLANT_ARCHITECT_API StrandsIntersection : public IPrivateComponent{
        friend class StrandPlant;
        std::vector<std::shared_ptr<StrandKnot>> m_strandKnots;

        bool m_isRoot = false;
        int m_maxDistanceToBoundary = 0;
        bool DisplayIntersection(const std::string& title, bool editable);
    public:
        void CleanChildren() const;
        [[nodiscard]] std::vector<std::shared_ptr<StrandKnot>> GetBoundaryKnots() const;
        void CalculateConnectivity();
        void OnCreate() override;
        void OnInspect() override;
        [[nodiscard]] bool CheckBoundary(const std::vector<glm::vec2>& points);
        void Construct(const std::vector<glm::vec2>& points);
        void Extract(const std::vector<SplitSettings> &targets,
                     std::vector<std::vector<std::shared_ptr<StrandKnot>>> &extractedKnots) const;
        [[nodiscard]] Entity Extend() const;
        [[nodiscard]] std::vector<Entity> Split(const std::vector<SplitSettings>& targets, const std::function<void(std::vector<std::shared_ptr<StrandKnot>> &srcList, std::vector<std::shared_ptr<StrandKnot>> &dstList)>& extendFunc);
    };

    class PLANT_ARCHITECT_API StrandPlant : public IPrivateComponent{
        std::vector<std::shared_ptr<Strand>> m_strands;
    public:
        [[nodiscard]] Entity GetRoot();
        void OnCreate() override;
        void OnInspect() override;
        void GenerateStrands();
    };
}