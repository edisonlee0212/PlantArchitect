#include <InternodeSystem.hpp>
#include <AutoTreeGenerationPipeline.hpp>
using namespace PlantArchitect;
namespace Scripts {
    class SpaceColonizationTreeToLString : public IAutoTreeGenerationPipelineBehaviour {
        int m_remainingInstanceAmount = 0;
        int m_remainingGrowthIterations = 0;
    public:
        int m_generationAmount = 10;
        std::filesystem::path m_currentExportFolder;
        int m_perTreeGrowthIteration = 40;
        PrivateComponentRef m_volume;
        int m_attractionPointAmount = 1000;
        void OnIdle(AutoTreeGenerationPipeline& pipeline) override;
        void OnBeforeGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnGrowth(AutoTreeGenerationPipeline& pipeline) override;
        void OnAfterGrowth(AutoTreeGenerationPipeline& pipeline) override;
    };
}