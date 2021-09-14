#pragma once
#include <InternodeSystem.hpp>
#include <AutoTreeGenerationPipeline.hpp>
#include <GeneralTreeBehaviour.hpp>
using namespace PlantArchitect;
namespace Scripts {
    class GeneralTreeToLString : public IAutoTreeGenerationPipelineBehaviour {
        int m_remainingInstanceAmount = 0;
        Entity m_currentGrowingTree;
        std::weak_ptr<GeneralTreeBehaviour> m_generalTreeBehaviour;
        bool m_imageCapturing = false;
    public:
        std::string m_parameterFileName;
        GeneralTreeParameters m_parameters;
        int m_generationAmount = 10;
        std::filesystem::path m_currentExportFolder = "./GeneralTreeToString_Export/";
        int m_perTreeGrowthIteration = 40;
        void OnIdle(AutoTreeGenerationPipeline& pipeline) override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;
    };
}