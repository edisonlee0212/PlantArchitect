#pragma once

#include <PlantLayer.hpp>
#include <AutoTreeGenerationPipeline.hpp>
#include "VoxelGrid.hpp"
#ifdef RAYTRACERFACILITY

#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"

using namespace RayTracerFacility;
#endif
using namespace PlantArchitect;
namespace Scripts {

    class TreeDataCapturePipeline : public IAutoTreeGenerationPipelineBehaviour {
        std::vector<glm::mat4> m_cameraModels;
        std::vector<glm::mat4> m_treeModels;
        std::vector<glm::mat4> m_projections;
        std::vector<glm::mat4> m_views;
        std::vector<std::string> m_names;
        Entity m_prefabEntity;
        Entity m_ground;
        Entity m_obstacle;

        AssetRef m_obstacleGrid;

        GlobalTransform TransformCamera(const Bound &bound, float turnAngle, float pitchAngle);

        void SetUpCamera(AutoTreeGenerationPipeline &pipeline);

        void ExportMatrices(const std::filesystem::path &path);

        void ExportGraphNode(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                             YAML::Emitter &out, int parentIndex, const Entity &internode);

        void ExportGraph(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                         const std::filesystem::path &path);

        void ExportCSV(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                       const std::filesystem::path &path);

        void ExportEnvironmentalGrid(AutoTreeGenerationPipeline &pipeline,
                                     const std::filesystem::path &path);
    public:
        EntityRef m_volumeEntity;
        MeshGeneratorSettings m_meshGeneratorSettings;
        bool m_enableRandomObstacle = false;
        bool m_renderObstacle = true;
        bool m_lShapedWall = false;
        glm::vec2 m_obstacleDistanceRange = glm::vec2(2, 10);
        glm::vec3 m_wallSize = glm::vec3(2.0f, 5.0f, 20.0f);
        bool m_randomRotation = true;
        AssetRef m_branchTexture;
        BehaviourType m_defaultBehaviourType = BehaviourType::GeneralTree;
#ifdef RAYTRACERFACILITY
        RayProperties m_rayProperties = {1, 512};
#endif
        AssetRef m_foliagePhyllotaxis;
        bool m_enableGround = true;
        bool m_autoAdjustCamera = true;
        bool m_applyPhyllotaxis = false;
        std::filesystem::path m_currentExportFolder;
        float m_branchWidth = 0.04f;
        float m_nodeSize = 0.05f;
            glm::vec3 m_focusPoint = glm::vec3(0, 15, 0);
        int m_pitchAngleStart = 0;
        int m_pitchAngleStep = 10;
        int m_pitchAngleEnd = 10;
        int m_turnAngleStart = 0.0f;
        int m_turnAngleStep = 360;
        int m_turnAngleEnd = 360.0f;
        float m_distance = 80;
        float m_fov = 60;
        float m_lightSize = 0.02f;
        float m_ambientLightIntensity = 0.15f;
        float m_envLightIntensity = 1.0f;
        glm::ivec2 m_resolution = glm::ivec2(1024, 1024);

        //Options.
        bool m_exportTreeIOTrees = false;
        bool m_exportOBJ = false;
        bool m_exportCSV = true;
        bool m_exportEnvironmentalGrid = false;
        bool m_exportWallPrefab = false;
        bool m_exportGraph = false;
        bool m_exportImage = false;
        bool m_exportDepth = false;
        bool m_exportMatrices = true;
        bool m_exportBranchCapture = false;
        bool m_exportLString = false;

        bool m_useClearColor = true;
        glm::vec3 m_backgroundColor = glm::vec3(1.0f);
        float m_cameraMax = 200;

        void OnStart(AutoTreeGenerationPipeline &pipeline) override;

        void OnEnd(AutoTreeGenerationPipeline &pipeline) override;

        void OnCreate() override;

        void OnBeforeGrowth(AutoTreeGenerationPipeline &pipeline) override;

        void OnAfterGrowth(AutoTreeGenerationPipeline &pipeline) override;

        void OnInspect() override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void DisableAllExport();
    };
}