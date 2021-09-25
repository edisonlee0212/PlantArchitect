#pragma once
#include <InternodeSystem.hpp>
#include <AutoTreeGenerationPipeline.hpp>
#include <GeneralTreeBehaviour.hpp>
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
        Entity m_currentGrowingTree;
        std::weak_ptr<GeneralTreeBehaviour> m_generalTreeBehaviour;
        bool m_skipCurrentFrame = false;

        std::vector<glm::mat4> m_cameraModels;
        std::vector<glm::mat4> m_treeModels;
        std::vector<glm::mat4> m_projections;
        std::vector<glm::mat4> m_views;
        std::vector<std::string> m_names;
        MultipleAngleCaptureStatus m_captureStatus = MultipleAngleCaptureStatus::Info;

        bool SetUpCamera();
        void RenderBranchCapture();
        void ExportMatrices(const std::filesystem::path& path);
        void ExportGraph(const std::filesystem::path& path);

        std::shared_ptr<Camera> m_branchCaptureCamera;
        glm::vec3 m_cameraPosition;
        glm::quat m_cameraRotation;
    public:
        std::string m_parameterFileName;
        GeneralTreeParameters m_parameters;
        int m_generationAmount = 2;
        std::filesystem::path m_currentExportFolder = "MultipleAngleCapture_Export/";
        int m_perTreeGrowthIteration = 40;

        float m_branchWidth = 0.04f;
        float m_nodeSize = 0.1f;
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
        bool m_exportBranchCapture = true;

        EntityRef m_cameraEntity;
        bool m_useClearColor = true;
        glm::vec3 m_backgroundColor = glm::vec3(1.0f);
        float m_cameraMin = 1;
        float m_cameraMax = 200;

        void OnCreate();
        void OnIdle(AutoTreeGenerationPipeline& pipeline) override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;
    };
}