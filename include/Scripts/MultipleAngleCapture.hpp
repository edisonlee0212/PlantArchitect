#pragma once
#include <InternodeSystem.hpp>
#include <AutoTreeGenerationPipeline.hpp>

using namespace PlantArchitect;
namespace Scripts {
    enum class MultipleAngleCaptureStatus{
        Info,
        Image
    };

    class MultipleAngleCapture : public IAutoTreeGenerationPipelineBehaviour {
        int m_pitchAngle = 0;
        int m_turnAngle = 0;
        int m_remainingInstanceAmount = 0;
        std::vector<glm::mat4> m_cameraModels;
        std::vector<glm::mat4> m_treeModels;
        std::vector<glm::mat4> m_projections;
        std::vector<glm::mat4> m_views;
        std::vector<std::string> m_names;
        MultipleAngleCaptureStatus m_captureStatus = MultipleAngleCaptureStatus::Info;
        bool SetUpCamera();
        void RenderBranchCapture();
        void ExportMatrices(const std::filesystem::path& path);
        void ExportGraph(const std::shared_ptr<IInternodeBehaviour>& behaviour, const std::filesystem::path& path);

        std::shared_ptr<Camera> m_branchCaptureCamera;
        glm::vec3 m_cameraPosition;
        glm::quat m_cameraRotation;
    public:
        bool m_autoAdjustCamera = true;
        int m_generationAmount = 2;
        PrivateComponentRef m_volume;
        std::filesystem::path m_currentExportFolder = "MultipleAngleCapture_Export/";
        int m_targetDivider = 512;
        float m_branchWidth = 0.04f;
        float m_nodeSize = 0.05f;
        glm::vec3 m_focusPoint = glm::vec3(0, 3, 0);
        float m_pitchAngleStart = 0;
        float m_pitchAngleStep = 10;
        float m_pitchAngleEnd = 30;
        float m_turnAngleStep = 90;
        float m_distance = 4.5;
        float m_fov = 60;
        glm::ivec2 m_resolution = glm::ivec2(1024, 1024);

        //Options.
        bool m_exportOBJ = false;
        bool m_exportGraph = true;
        bool m_exportImage = true;
        bool m_exportDepth = true;
        bool m_exportMatrices = true;
        bool m_exportBranchCapture = true;
        bool m_exportLString = true;
        EntityRef m_cameraEntity;
        bool m_useClearColor = true;
        glm::vec3 m_backgroundColor = glm::vec3(1.0f);
        float m_cameraMin = 1;
        float m_cameraMax = 200;

        void OnCreate() override;
        void OnIdle(AutoTreeGenerationPipeline& pipeline) override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;
    };
}