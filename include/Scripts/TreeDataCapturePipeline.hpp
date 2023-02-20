#pragma once

#include "InternodeModel/InternodeLayer.hpp"
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

        void SetUpCamera(AutoTreeGenerationPipeline &pipeline);

        void ExportMatrices(const std::filesystem::path &path);

        void ExportGraphNode(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                             YAML::Emitter &out, int parentIndex, const Entity &internode);

        void ExportGraph(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                         const std::filesystem::path &path, int treeIndex);

        void ExportCSV(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                       const std::filesystem::path &path, int treeIndex);

        void ExportEnvironmentalGrid(AutoTreeGenerationPipeline &pipeline,
                                     const std::filesystem::path &path);

        void ExportJunction(AutoTreeGenerationPipeline &pipeline, const std::shared_ptr<IPlantBehaviour> &behaviour,
                                     const std::filesystem::path &path);

        std::vector<std::pair<std::string, std::string>> m_treeIOPairs;

    public:
        bool m_enableMultipleTrees = false;
        std::vector<glm::vec3> m_treePositions;

        AssetRef m_obstacleGrid;
        EntityRef m_volumeEntity;
        std::filesystem::path m_currentExportFolder;
        MeshGeneratorSettings m_meshGeneratorSettings;
        BehaviourType m_defaultBehaviourType = BehaviourType::GeneralTree;

        struct ObstacleSettings {
            bool m_enableRandomObstacle = false;
            bool m_renderObstacle = true;
            bool m_lShapedWall = false;
            glm::vec2 m_obstacleDistanceRange = glm::vec2(2, 10);
            glm::vec3 m_wallSize = glm::vec3(2.0f, 5.0f, 20.0f);
            bool m_randomRotation = true;
        } m_obstacleSettings;

        struct AppearanceSettings {
            AssetRef m_branchTexture;

            AssetRef m_foliagePhyllotaxis;
            bool m_applyPhyllotaxis = false;
            float m_branchWidth = 0.04f;
            float m_nodeSize = 0.05f;
        } m_appearanceSettings;

        struct EnvironmentSettings {
            bool m_enableGround = true;
            float m_lightSize = 0.02f;
            float m_ambientLightIntensity = 0.15f;
            float m_envLightIntensity = 1.0f;
        } m_environmentSettings;

        struct CameraCaptureSettings {
#ifdef RAYTRACERFACILITY
            RayProperties m_rayProperties = {1, 512};
#endif
            glm::vec3 m_focusPoint = glm::vec3(0, 15, 0);
            bool m_autoAdjustFocusPoint = true;
            int m_pitchAngleStart = 0;
            int m_pitchAngleStep = 10;
            int m_pitchAngleEnd = 10;
            int m_turnAngleStart = 0.0f;
            int m_turnAngleStep = 360;
            int m_turnAngleEnd = 360.0f;
            float m_distance = 80;
            float m_fov = 60;
            glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
            bool m_useClearColor = true;
            glm::vec3 m_backgroundColor = glm::vec3(1.0f);
            float m_cameraDepthMax = 200;

            void OnInspect();

            void Serialize(const std::string &name, YAML::Emitter &out);

            void Deserialize(const std::string &name, const YAML::Node &in);

            GlobalTransform GetTransform(bool isCamera, const Bound &bound, float turnAngle, float pitchAngle);
        } m_cameraSettings, m_pointCloudSettings;

        struct PointCloudPointSettings {
            bool m_color = false;
            bool m_pointType = true;
            float m_variance = 0.15f;
            float m_ballRandRadius = 0.0f;
            bool m_junction = false;
            bool m_internodeIndex = false;
            float m_boundingBoxOffset = 100.0f;
        } m_pointCloudPointSettings;
#ifdef RAYTRACERFACILITY
        void ScanPointCloudLabeled(const Bound &plantBound, AutoTreeGenerationPipeline &pipeline,
                                   const std::filesystem::path &savePath);
#endif
        struct ExportOptions {
            bool m_exportTreeIOTrees = false;
            bool m_exportOBJ = false;
            bool m_exportCSV = true;
            bool m_exportEnvironmentalGrid = false;
            bool m_exportWallPrefab = false;
            bool m_exportGraph = false;
            bool m_exportMask = false;
            bool m_exportRBV = false;
            bool m_exportImage = false;
            bool m_exportDepth = false;
            bool m_exportMatrices = true;
            bool m_exportBranchCapture = false;
            bool m_exportLString = false;
            bool m_exportPointCloud = false;
        } m_exportOptions;

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