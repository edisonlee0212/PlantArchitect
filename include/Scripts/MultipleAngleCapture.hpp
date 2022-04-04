#pragma once
#include <PlantLayer.hpp>
#include <AutoTreeGenerationPipeline.hpp>
#ifdef RAYTRACERFACILITY
#include "RayTracerCamera.hpp"
#include "RayTracerLayer.hpp"
using namespace RayTracerFacility;
#endif
using namespace PlantArchitect;
namespace Scripts {
    class MultipleAngleCapture : public IAutoTreeGenerationPipelineBehaviour {
        std::vector<glm::mat4> m_cameraModels;
        std::vector<glm::mat4> m_treeModels;
        std::vector<glm::mat4> m_projections;
        std::vector<glm::mat4> m_views;
        std::vector<std::string> m_names;
        GlobalTransform TransformCamera(const Bound& bound, float turnAngle, float pitchAngle);
        void SetUpCamera(AutoTreeGenerationPipeline& pipeline);
        void ExportMatrices(const std::filesystem::path& path);
        void ExportGraphNode(const std::shared_ptr<IPlantBehaviour>& behaviour, YAML::Emitter &out, int parentIndex, const Entity& internode);
        void ExportGraph(AutoTreeGenerationPipeline& pipeline, const std::shared_ptr<IPlantBehaviour>& behaviour, const std::filesystem::path& path);
        void ExportCSV(AutoTreeGenerationPipeline& pipeline, const std::shared_ptr<IPlantBehaviour>& behaviour, const std::filesystem::path& path);
    public:
        AssetRef m_foliageTexture;
        AssetRef m_branchTexture;
        BehaviourType m_defaultBehaviourType = BehaviourType::GeneralTree;
#ifdef RAYTRACERFACILITY
        RayProperties m_rayProperties = {1, 128};
#endif
        AssetRef m_foliagePhyllotaxis;
        bool m_autoAdjustCamera = true;
        bool m_applyPhyllotaxis = false;
        std::filesystem::path m_currentExportFolder;
        float m_branchWidth = 0.04f;
        float m_nodeSize = 0.05f;
        glm::vec3 m_focusPoint = glm::vec3(0, 3, 0);
        int m_pitchAngleStart = 0;
        int m_pitchAngleStep = 10;
        int m_pitchAngleEnd = 10;
        int m_turnAngleStart = 0.0f;
        int m_turnAngleStep = 360;
        int m_turnAngleEnd = 360.0f;
        float m_distance = 4.5;
        float m_fov = 60;
        float m_lightSize = 0.02f;
        float m_ambientLightIntensity = 0.1f;
        float m_envLightIntensity = 1.0f;
        glm::ivec2 m_resolution = glm::ivec2(1024, 1024);

        //Options.
        bool m_exportTreeIOTrees = false;
        bool m_exportOBJ = false;
        bool m_exportCSV = true;
        bool m_exportGraph = false;
        bool m_exportImage = false;
        bool m_exportDepth = false;
        bool m_exportMatrices = true;
        bool m_exportBranchCapture = false;
        bool m_exportLString = false;

        bool m_useClearColor = true;
        glm::vec3 m_backgroundColor = glm::vec3(1.0f);
        float m_cameraMax = 200;
        void OnStart(AutoTreeGenerationPipeline& pipeline) override;
        void OnEnd(AutoTreeGenerationPipeline& pipeline) override;
        void OnCreate() override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;

        void DisableAllExport();
        ~MultipleAngleCapture();
    };
}