#pragma once

#include <plant_architect_export.h>

#include <CUDAModule.hpp>
#include <Camera.hpp>
#include <InternodeRingSegment.hpp>
#include <TreeData.hpp>
#include <Volume.hpp>
#include <VoxelSpace.hpp>
#include <QuickHull.hpp>

using namespace UniEngine;
namespace PlantArchitect {
    enum class PLANT_ARCHITECT_API PlantType {
        GeneralTree, Sorghum
    };
#pragma region Tree
    struct PLANT_ARCHITECT_API PlantInfo : IDataComponent {
        PlantType m_plantType;
        float m_startTime;
        float m_age;
    };
#pragma endregion
#pragma region Internode

    struct PLANT_ARCHITECT_API BranchCylinder : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API BranchCylinderWidth : IDataComponent {
        float m_value;

        bool operator==(const BranchCylinderWidth &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API BranchPointer : IDataComponent {
        glm::mat4 m_value;

        bool operator==(const BranchCylinder &other) const {
            return other.m_value == m_value;
        }
    };

    struct PLANT_ARCHITECT_API Illumination : IDataComponent {
        float m_currentIntensity = 0;
        glm::vec3 m_accumulatedDirection = glm::vec3(0.0f);
    };

    struct PLANT_ARCHITECT_API BranchColor : IDataComponent {
        glm::vec4 m_value;
    };

    struct PLANT_ARCHITECT_API InternodeInfo : IDataComponent {
        PlantType m_plantType;

        float m_usedResource;
        bool m_activated = true;
        float m_startAge = 0;
        float m_startGlobalTime = 0;
        int m_order = 1;
        int m_level = 1;
        int m_index = -1;
    };
    struct PLANT_ARCHITECT_API InternodeGrowth : IDataComponent {
        float m_inhibitor = 0;
        float m_inhibitorTransmitFactor = 1;
        int m_distanceToRoot = 0; // Ok
        float m_internodeLength = 0.0f;

        glm::vec3 m_branchEndPosition = glm::vec3(0);
        glm::vec3 m_branchStartPosition = glm::vec3(0);
        glm::vec3 m_childrenTotalTorque = glm::vec3(0.0f);
        glm::vec3 m_childMeanPosition = glm::vec3(0.0f);
        float m_MassOfChildren = 0;
        float m_sagging = 0.0f;

        float m_thickness = 0.02f; // Ok
        glm::quat m_desiredLocalRotation = glm::quat(glm::vec3(0.0f));
        glm::quat m_desiredGlobalRotation = glm::quat(glm::vec3(0.0f));
        // Will be used to calculate the gravity bending.
        glm::vec3 m_desiredGlobalPosition = glm::vec3(0.0f);

    };
    struct PLANT_ARCHITECT_API InternodeStatistics : IDataComponent {
        int m_childrenEndNodeAmount = 0;       // Ok
        bool m_isMaxChild = false;             // Ok
        bool m_isEndNode = false;              // Ok
        int m_maxChildOrder = 0;               // Ok
        int m_maxChildLevel = 0;               // Ok
        int m_distanceToBranchEnd = 0;         // Ok The no-branching chain distance.
        int m_longestDistanceToAnyEndNode = 0; // Ok
        int m_totalLength = 0;                 // Ok
        int m_distanceToBranchStart = 0;       // Ok
    };

#pragma endregion
#pragma region Bud

    class Bud;

    struct PLANT_ARCHITECT_API InternodeCandidate {
        Entity m_plant;
        Entity m_parent;
        std::vector<Bud> m_buds;
        GlobalTransform m_globalTransform;
        InternodeInfo m_info = InternodeInfo();
        InternodeGrowth m_growth = InternodeGrowth();
        InternodeStatistics m_statistics = InternodeStatistics();
    };

    struct PLANT_ARCHITECT_API ResourceParcel {
        float m_nutrient;
        float m_carbon;
        float m_globalTime = 0;

        ResourceParcel();

        ResourceParcel(const float &water, const float &carbon);

        ResourceParcel &operator+=(const ResourceParcel &value);

        [[nodiscard]] bool IsEnough() const;

        void OnGui() const;

        void Serialize(YAML::Emitter &out);

        void Deserialize(const YAML::Node &in);

    };

    class PLANT_ARCHITECT_API Bud {
    public:
        bool m_enoughForGrowth = false;
        float m_resourceWeight = 1.0f;
        ResourceParcel m_currentResource;
        std::vector<ResourceParcel> m_resourceLog;
        float m_deathGlobalTime = -1;
        float m_avoidanceAngle;
        bool m_active = true;
        bool m_isApical = false;
        float m_mainAngle = 0;

        void Serialize(YAML::Emitter &out);

        void Deserialize(const YAML::Node &in);
    };

#pragma endregion

    struct PLANT_ARCHITECT_API KDop {
        std::vector<Plane> m_planes;
        std::vector<Vertex> m_vertices;
        std::vector<glm::uvec3> m_indices;

        void Serialize(YAML::Emitter &out);

        void Deserialize(const YAML::Node &in);

        void Calculate(const std::vector<glm::mat4> &globalTransforms);

        glm::vec3 GetIntersection(const Plane &p0, const Plane &p1, const Plane &p2);
    };

    class PLANT_ARCHITECT_API InternodeData : public IPrivateComponent {
    public:
        glm::vec3 m_normalDir = glm::vec3(0, 0, 1);
        bool m_displayPoints = true;
        bool m_displayHullMesh = true;
        float m_pointSize = 0.001f;
        float m_lineWidth = 5.0f;
        glm::vec4 m_pointColor = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
        glm::vec4 m_hullMeshColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);


        EntityRef m_thickestChild;
        EntityRef m_plant;
        std::vector<glm::mat4> m_leavesTransforms;
        std::vector<glm::mat4> m_points;
        std::vector<Bud> m_buds;
        std::vector<InternodeRingSegment> m_rings;
        KDop m_kDop;
        quickhull::ConvexHull<float> m_convexHull;
        std::shared_ptr<Mesh> m_hullMesh;


        int m_step;

        void OnGui() override;

        void CalculateKDop();

        void CalculateQuickHull();

        void FormMesh();

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void Relink(const std::unordered_map<Handle, Handle> &map) override;

        void Clone(const std::shared_ptr<IPrivateComponent> &target) override;
    };

#pragma region Enums
    enum class PLANT_ARCHITECT_API BranchRenderType {
        Illumination,
        Sagging,
        Inhibitor,
        InhibitorTransmitFactor,
        ResourceToGrow,
        Order,
        MaxChildOrder,
        Level,
        MaxChildLevel,
        IsMaxChild,
        ChildrenEndNodeAmount,
        IsEndNode,
        DistanceToBranchEnd,
        DistanceToBranchStart,
        TotalLength,
        LongestDistanceToAnyEndNode,

    };

    enum class PLANT_ARCHITECT_API PointerRenderType {
        Illumination, Bending
    };
#pragma endregion

    class PLANT_ARCHITECT_API PlantSystem : public ISystem {
    public:
        std::map<PlantType,
                std::function<void(std::vector<ResourceParcel> &resources)>>
                m_plantResourceAllocators;
        std::map<PlantType,
                std::function<void(std::vector<InternodeCandidate> &candidates)>>
                m_plantGrowthModels;
        std::map<PlantType,
                std::function<void(
                        std::vector<std::pair<GlobalTransform, Volume *>> &obstacles)>>
                m_plantInternodePruners;
        std::map<PlantType, std::function<void(const Entity &newInternode,
                                               const InternodeCandidate &candidate)>>
                m_plantInternodePostProcessors;

        std::map<PlantType, std::function<void()>> m_plantMetaDataCalculators;
        std::map<PlantType, std::function<void()>> m_plantMeshGenerators;
        std::map<PlantType, std::function<void()>> m_plantSkinnedMeshGenerators;
        std::map<PlantType, std::function<void()>> m_deleteAllPlants;
#pragma region Growth

        bool m_needUpdateMetadata = false;

        bool GrowAllPlants();

        bool GrowAllPlants(const unsigned &iterations);

        bool GrowCandidates(std::vector<InternodeCandidate> &candidates);

        void CalculateIlluminationForInternodes();

        void CollectNutrient(std::vector<Entity> &trees,
                             std::vector<ResourceParcel> &totalNutrients,
                             std::vector<ResourceParcel> &nutrientsAvailable);

        void ApplyTropism(const glm::vec3 &targetDir, float tropism, glm::vec3 &front,
                          glm::vec3 &up);

#pragma endregion
#pragma region Members
        EntityRef m_ground;
        EntityRef m_anchor;
        /**
         * \brief The period of time for each iteration. Must be smaller than 1.0f.
         */
        float m_deltaTime = 1.0f;
        /**
         * \brief The current global time.
         */
        float m_globalTime;
        /**
         * \brief Whether the PlanetManager is initialized.
         */
        bool m_ready;
        float m_illuminationFactor = 0.002f;
        float m_illuminationAngleFactor = 2.0f;

        float m_physicsTimeStep = 1.0f / 60;
        float m_physicsSimulationTotalTime = 0.0f;
        float m_physicsSimulationRemainingTime = 0.0f;

        EntityArchetype m_internodeArchetype;
        EntityArchetype m_plantArchetype;
        EntityQuery m_plantQuery;
        EntityQuery m_internodeQuery;

        std::vector<Entity> m_plants;
        std::vector<Entity> m_internodes;
        std::vector<GlobalTransform> m_internodeTransforms;

#pragma region Timers
        float m_meshGenerationTimer = 0;
        float m_resourceAllocationTimer = 0;
        float m_internodeFormTimer = 0;
        float m_internodeCreateTimer = 0;
        float m_internodeCreatePostProcessTimer = 0;
        float m_illuminationCalculationTimer = 0;
        float m_pruningTimer = 0;
        float m_metaDataTimer = 0;
#pragma endregion
#pragma region GUI Settings
        int m_iterationsToGrow = 0;
        bool m_endUpdate = false;

#pragma endregion

#pragma endregion
#pragma region Helpers

        Entity CreateCubeObstacle();

        void DeleteAllPlants();

        Entity CreatePlant(const PlantType &type, const Transform &transform);

        Entity CreateInternode(const PlantType &type, const Entity &parentEntity);

#pragma endregion
#pragma region Runtime

        void OnInspect() override;

        void Update() override;

        void Refresh();

        void End();

        void OnCreate() override;

        void Start() override;

        void PhysicsSimulate();

#pragma endregion

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void Relink(const std::unordered_map<Handle, Handle> &map);
    };
} // namespace PlantFactory
