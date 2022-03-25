#pragma once
#include <PlantLayer.hpp>
#include <AutoTreeGenerationPipeline.hpp>

using namespace PlantArchitect;
namespace Scripts {
    class LSystemStringToImages : public IAutoTreeGenerationPipelineBehaviour {
        Entity m_scene;
    public:
        Entity m_rayTracerCamera;
        std::filesystem::path m_currentExportFolder = "Datasets/";
        AssetRef m_prefab;
    };
}