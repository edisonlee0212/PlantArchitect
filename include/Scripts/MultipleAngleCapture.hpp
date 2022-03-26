#pragma once
#include <PlantLayer.hpp>
#include <AutoTreeGenerationPipeline.hpp>

#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
using namespace PlantArchitect;
namespace Scripts {
    class MultipleAngleCapture : public IAutoTreeGenerationPipelineBehaviour {
        std::vector<glm::mat4> m_cameraModels;
        std::vector<glm::mat4> m_treeModels;
        std::vector<glm::mat4> m_projections;
        std::vector<glm::mat4> m_views;
        std::vector<std::string> m_names;
        void SetUpCamera(AutoTreeGenerationPipeline& pipeline);
        void ExportMatrices(const std::filesystem::path& path);
        void ExportGraphNode(const std::shared_ptr<IPlantBehaviour>& behaviour, YAML::Emitter &out, int parentIndex, const Entity& internode);
        void ExportGraph(const std::shared_ptr<IPlantBehaviour>& behaviour, const std::filesystem::path& path);

        void ExportCSV(const std::shared_ptr<IPlantBehaviour>& behaviour, const std::filesystem::path& path);

        glm::vec3 m_cameraPosition;
        glm::quat m_cameraRotation;
    public:
        RayProperties m_rayProperties = {1, 1};
        AssetRef m_foliagePhyllotaxis;
        bool m_autoAdjustCamera = true;

        PrivateComponentRef m_volume;
        std::filesystem::path m_currentExportFolder = "Datasets/";
        float m_branchWidth = 0.04f;
        float m_nodeSize = 0.05f;
        glm::vec3 m_focusPoint = glm::vec3(0, 3, 0);
        float m_pitchAngleStart = 0;
        float m_pitchAngleStep = 10;
        float m_pitchAngleEnd = 10;
        float m_turnAngleStart = 0.0f;
        float m_turnAngleStep = 360;
        float m_turnAngleEnd = 360.0f;
        float m_distance = 4.5;
        float m_fov = 60;
        glm::ivec2 m_resolution = glm::ivec2(1024, 1024);
        int m_startIndex = 0;
        //Options.
        BranchColorMode m_branchColorMode = BranchColorMode::None;
        bool m_exportOBJ = false;
        bool m_exportCSV = true;
        bool m_exportGraph = false;
        bool m_exportImage = false;
        bool m_exportDepth = false;
        bool m_exportMatrices = false;
        bool m_exportBranchCapture = false;
        bool m_exportLString = false;

        bool m_useClearColor = true;
        glm::vec3 m_backgroundColor = glm::vec3(1.0f);
        float m_cameraMin = 1;
        float m_cameraMax = 300;
        void Start(AutoTreeGenerationPipeline& pipeline) override;
        void OnCreate() override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;

        void DisableAllExport();
    };
}