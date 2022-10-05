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

        glm::vec3 m_position;
        glm::vec3 m_direction;
        float m_thickness;
        glm::vec3 m_color = glm::vec3(1.0f);

    };

    class PLANT_ARCHITECT_API Strand {
    public:
        std::weak_ptr<StrandKnot> m_start;
        void BuildStrands(std::vector<int> &strands,
                          std::vector<StrandPoint> &points);
    };

    struct SplitSettings {
        int m_knotSize;
        glm::vec2 m_direction;
    };

    class PLANT_ARCHITECT_API StrandsIntersection {
        friend class StrandPlant;

        std::vector<std::shared_ptr<StrandKnot>> m_strandKnots;
        float m_unitDistance = 0.01f;
        bool m_isRoot = false;
        int m_maxDistanceToBoundary = 0;


        std::weak_ptr<StrandsIntersection> m_parent;
        std::vector<std::shared_ptr<StrandsIntersection>> m_children;
    public:
        std::string m_name = "New Intersection";
        Handle m_handle;
        Transform m_transform;
        [[nodiscard]] std::vector<std::shared_ptr<StrandKnot>> GetBoundaryKnots() const;
        void CalculateConnectivity();
        [[nodiscard]] bool CheckBoundary(const std::vector<glm::vec2> &points);
        void Construct(const std::vector<glm::vec2> &points);



        void CalculatePosition(const GlobalTransform &parentGlobalTransform) const;
    };

    class PLANT_ARCHITECT_API StrandPlant : public IPrivateComponent {
        std::vector<std::shared_ptr<Strand>> m_strands;
        std::vector<std::weak_ptr<StrandsIntersection>> m_selectedIntersectionHierarchyList;
        std::weak_ptr<StrandsIntersection> m_selectedStrandIntersection;
        bool DrawIntersectionMenu(const std::shared_ptr<StrandsIntersection>& strandIntersection);
        void SetSelectedIntersection(const std::shared_ptr<StrandsIntersection>& strandIntersection, bool openMenu);
        bool DisplayIntersection(const std::shared_ptr<StrandsIntersection>& strandIntersection, const std::string &title, bool editable);
        void DrawIntersectionGui(const std::shared_ptr<StrandsIntersection>& strandIntersection, bool& deleted, const unsigned &hierarchyLevel);
    public:
        std::shared_ptr<StrandsIntersection> m_root;

        void Extract(const std::shared_ptr<StrandsIntersection>& strandIntersection, const std::vector<SplitSettings> &targets,
                     std::vector<std::vector<std::shared_ptr<StrandKnot>>> &extractedKnots) const;
        void Extend(const std::shared_ptr<StrandsIntersection>& strandIntersection);
        void Split(const std::shared_ptr<StrandsIntersection>& strandIntersection, const std::vector<SplitSettings> &targets,
                   const std::function<void(
                           std::vector<std::shared_ptr<StrandKnot>> &srcList,
                           std::vector<std::shared_ptr<StrandKnot>> &dstList
                   )> &extendFunc);

        void OnCreate() override;

        void OnInspect() override;

        void GenerateStrands();

        void InitializeStrandRenderer() const;
    };
}