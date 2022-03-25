#pragma once
#include <PlantLayer.hpp>
#include <GeneralTreeBehaviour.hpp>
#include <SpaceColonizationBehaviour.hpp>
#include <LSystemBehaviour.hpp>
using namespace PlantArchitect;
namespace Scripts {
    class GANTreePipelineDriver : public IPrivateComponent {
    public:
        int m_instancePerSpecie = 2;
        std::filesystem::path m_folderPath = "";
        PrivateComponentRef m_pipeline;
        std::vector<AssetRef> m_descriptors;
        void LateUpdate() override;
        void OnInspect() override;
        void Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) override;
    };
}