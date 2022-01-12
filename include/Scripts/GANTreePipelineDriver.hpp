#pragma once
#include <InternodeLayer.hpp>
#include <GeneralTreeBehaviour.hpp>
#include <SpaceColonizationBehaviour.hpp>
#include <LSystemBehaviour.hpp>
using namespace PlantArchitect;
namespace Scripts {
    class GANTreePipelineDriver : public IPrivateComponent {
        int m_age = 1;
    public:
        int m_instancePerSpecie = 2;
        std::filesystem::path m_folderPath = "C:\\Users\\lllll\\Documents\\GitHub\\PlantArchitect\\Resources\\Parameters\\";
        PrivateComponentRef m_pipeline;
        std::vector<std::filesystem::path> m_parameterFilePaths;
        void LateUpdate() override;
        void OnInspect() override;
        void Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) override;
    };
}