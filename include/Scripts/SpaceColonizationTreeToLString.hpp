#pragma once
#include <InternodeSystem.hpp>
#include <AutoTreeGenerationPipeline.hpp>
#include <SpaceColonizationBehaviour.hpp>
using namespace PlantArchitect;
namespace Scripts {
    class SpaceColonizationTreeToLString : public IAutoTreeGenerationPipelineBehaviour {
        int m_remainingInstanceAmount = 0;
        Entity m_currentGrowingTree;
        std::weak_ptr<SpaceColonizationBehaviour> m_spaceColonizationTreeBehaviour;
        bool m_imageCapturing = false;
    public:
        SpaceColonizationParameters m_parameters;
        int m_generationAmount = 10;
        std::filesystem::path m_currentExportFolder = "./SpaceColonizationTreeToString_Export/";
        int m_perTreeGrowthIteration = 40;
        int m_attractionPointAmount = 8000;
        void OnIdle(AutoTreeGenerationPipeline& pipeline) override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;


    };
}