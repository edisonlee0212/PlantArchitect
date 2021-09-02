#include <InternodeSystem.hpp>
#include <AutoTreeGenerationPipeline.hpp>
#include <SpaceColonizationBehaviour.hpp>
using namespace PlantArchitect;
namespace Scripts {
    class SpaceColonizationTreeToLString : public IAutoTreeGenerationPipelineBehaviour {
        int m_remainingInstanceAmount = 0;
        int m_remainingGrowthIterations = 0;
        Entity m_currentGrowingTree;
        std::weak_ptr<SpaceColonizationBehaviour> m_spaceColonizationTreeBehaviour;
    public:
        int m_generationAmount = 10;
        std::filesystem::path m_currentExportFolder = "./export/";
        int m_perTreeGrowthIteration = 40;
        PrivateComponentRef m_volume;
        int m_attractionPointAmount = 1000;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnInspect() override;
    };
}