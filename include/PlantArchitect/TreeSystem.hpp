#pragma once

#include <PlantSystem.hpp>
#include <rapidxml.hpp>
#include <rapidxml_print.hpp>
#include <rapidxml_utils.hpp>
#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    enum class PLANT_ARCHITECT_API GrowthParameterType {
        Gravitropism,
        Phototropism,
        InhibitorTransmitFactor,
        BranchingAngle,
        RollAngle,
        ApicalAngle,
        LateralBudPerNode,
        ResourceWeightApical,
        ResourceWeightLateral,
        AvoidanceAngle
    };

    struct PLANT_ARCHITECT_API TreeLeavesTag : IDataComponent {
    };

    struct PLANT_ARCHITECT_API RbvTag : IDataComponent {
    };

    class PLANT_ARCHITECT_API TreeSystem : public ISystem {
        SystemRef m_plantSystem;
        bool m_ready = false;
#pragma region Helpers

        static void ExportChains(int parentOrder, Entity internode,
                                 rapidxml::xml_node<> *chains,
                                 rapidxml::xml_document<> *doc);

        static void WriteChain(int order, Entity internode,
                               rapidxml::xml_node<> *chains,
                               rapidxml::xml_document<> *doc);

        Entity GetRootInternode(const Entity &tree);

        Entity GetLeaves(const Entity &tree);

        Entity GetRbv(const Entity &tree);

#pragma endregion
    public:
#pragma region Physics
        float m_density = 1.0f;
        float m_linearDamping = 2.0f;
        float m_angularDamping = 2.0f;
        int m_positionSolverIteration = 32;
        int m_velocitySolverIteration = 8;
        float m_jointDriveStiffnessFactor = 3000.0f;
        float m_jointDriveStiffnessThicknessFactor = 4.0f;
        float m_jointDriveDampingFactor = 10.0f;
        float m_jointDriveDampingThicknessFactor = 4.0f;
        bool m_enableAccelerationForDrive = true;
#pragma endregion

#pragma region Rendering
        VoxelSpace m_voxelSpaceModule;

        /**
         * \brief The information displayed when rendering internodes.
         */
        BranchRenderType m_branchRenderType = BranchRenderType::Order;
        PointerRenderType m_pointerRenderType = PointerRenderType::Illumination;
        float m_displayTime = 0;
        float m_previousGlobalTime = 0;
        float m_connectionWidth = 0.03f;
        bool m_displayThickness = true;

        bool m_updateBranch = false;
        bool m_updatePointer = false;
        bool m_alwaysUpdate = true;
        float m_pointerLength = 0.4f;
        float m_pointerWidth = 0.02f;

        bool m_drawVoxels = false;
        bool m_drawBranches = true;
        bool m_drawPointers = false;

        std::vector<glm::vec3> m_randomColors;
        float m_transparency = 0.7f;
        bool m_useTransparency = true;
        bool m_useColorMap = true;
        bool m_enableBranchDataCompress = false;
        float m_branchCompressFactor = 1.0f;

        bool m_enablePointerDataCompress = false;
        float m_pointerCompressFactor = 1.0f;

        int m_colorMapSegmentAmount = 3;
        std::vector<float> m_colorMapValues;
        std::vector<glm::vec3> m_colorMapColors;

        glm::vec4 m_pointerColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.5f);

        std::vector<Entity> m_entitiesWithRenderer;
        OpenGLUtils::GLVBO m_internodeColorBuffer;
#pragma endregion
#pragma region Crown shyness
        float m_crownShynessDiameter = 0.2f;
#pragma endregion

#pragma region Leaf Gen
        int m_leafAmount = 20;
        float m_radius = 0.3f;
        glm::vec2 m_leafSize = glm::vec2(0.05f);
        float m_distanceToEndNode = 0.5f;
#pragma endregion

#pragma region Internode debugging camera
        std::shared_ptr<Camera> m_internodeDebuggingCamera;
        int m_internodeDebuggingCameraResolutionX = 1;
        int m_internodeDebuggingCameraResolutionY = 1;
        float m_lastX = 0;
        float m_lastY = 0;
        float m_lastScrollY = 0;
        bool m_startMouse = false;
        bool m_startScroll = false;
        bool m_rightMouseButtonHold = false;

        EntityRef m_currentFocusingInternode = Entity();

#pragma endregion
        float m_meshResolution = 0.02f;
        float m_meshSubdivision = 4.0f;

        EntityArchetype m_leavesArchetype;
        EntityArchetype m_rbvArchetype;
        AssetRef m_defaultRayTracingBranchAlbedoTexture;
        AssetRef m_defaultRayTracingBranchNormalTexture;
        AssetRef m_defaultBranchAlbedoTexture;
        AssetRef m_defaultBranchNormalTexture;

        void Update() override;

        void LateUpdate() override;

        Entity CreateTree(const Transform &transform);

        void OnInspect() override;

        void InternodePostProcessor(const Entity &newInternode, const InternodeCandidate &candidate);

        void UpdateBranchCylinder(const bool &displayThickness,
                                  const float &width = 0.01f);

        void UpdateBranchPointer(const float &length,
                                 const float &width = 0.01f);

        void UpdateBranchColors();

        void ColorSet(glm::vec4 &target, const float &value);

        void RenderBranchCylinders(const float &displayTime);

        void RenderBranchPointers(const float &displayTime);

        void TreeNodeWalker(std::vector<Entity> &boundEntities,
                            std::vector<int> &parentIndices,
                            const int &parentIndex, const Entity &node);

        void TreeMeshGenerator(std::vector<Entity> &internodes,
                               std::vector<int> &parentIndices,
                               std::vector<Vertex> &vertices,
                               std::vector<unsigned> &indices);

        void TreeSkinnedMeshGenerator(std::vector<Entity> &internodes,
                                      std::vector<int> &parentIndices,
                                      std::vector<SkinnedVertex> &vertices,
                                      std::vector<unsigned> &indices);

        void GenerateMeshForTree();

        void GenerateSkinnedMeshForTree();

        void FormCandidates(std::vector<InternodeCandidate> &candidates);

        float GetGrowthParameter(const GrowthParameterType &type,
                                 const std::shared_ptr<TreeData> &treeData,
                                 InternodeInfo &internodeInfo,
                                 InternodeGrowth &internodeGrowth,
                                 InternodeStatistics &internodeStatistics);

        void PruneTrees(std::vector<std::pair<GlobalTransform, Volume *>> &obstacles);

        void UpdateTreesMetaData();

#pragma region Metadata

        void UpdateDistances(const Entity &internode, const std::shared_ptr<TreeData> &treeData);

        void UpdateLevels(const Entity &internode, const std::shared_ptr<TreeData> &treeData);

#pragma endregion

        void ResetTimeForTree(const float &value);

        void ResetTimeForTree(const Entity &internode,
                              const float &globalTime);

        void
        DistributeResourcesForTree(std::vector<ResourceParcel> &totalNutrients);

        void OnCreate() override;

        void Start() override;

        void SerializeScene(const std::string &filename);

        static void Serialize(const Entity &treeEntity, rapidxml::xml_document<> &doc,
                              rapidxml::xml_node<> *sceneNode);

        void DeleteAllPlantsHelper();


        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;

    };
} // namespace PlantFactory