//
// Created by lllll on 8/31/2021.
//

#include "LSystemBehaviour.hpp"
#include "InternodeSystem.hpp"
#include "EmptyInternodeResource.hpp"
void PlantArchitect::LSystemBehaviour::OnInspect() {
    CreateInternodeMenu<LSystemParameters>
            ("New Space Colonization Plant Wizard",
             ".scparams",
             [](LSystemParameters &params) {
                 params.OnInspect();
             },
             [](LSystemParameters &params,
                const std::filesystem::path &path) {

             },
             [](const LSystemParameters &params,
                const std::filesystem::path &path) {

             },
             [&](const LSystemParameters &params,
                 const Transform &transform) {
                 auto entity = Retrieve<EmptyInternodeResource>();
                 Transform internodeTransform;
                 internodeTransform.m_value =
                         glm::translate(glm::vec3(0.0f)) *
                         glm::mat4_cast(glm::quat(glm::vec3(glm::radians(90.0f), 0.0f, 0.0f))) *
                         glm::scale(glm::vec3(1.0f));
                 internodeTransform.m_value = transform.m_value * internodeTransform.m_value;
                 entity.SetDataComponent(internodeTransform);
                 LSystemTag tag;
                 entity.SetDataComponent(tag);
                 InternodeInfo newInfo;
                 newInfo.m_length = params.m_internodeLength;
                 newInfo.m_thickness = params.m_internodeLength / 10.0f;
                 entity.SetDataComponent(newInfo);
                 entity.SetDataComponent(params);
                 return entity;
             }
            );

    static float resolution = 0.02;
    static float subdivision = 4.0;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate branch mesh")) {
        GenerateBranchSkinnedMeshes(m_internodesQuery, subdivision, resolution);
    }

}

void PlantArchitect::LSystemBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Space Colonization Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("LSystem Internode", InternodeInfo(),
                                                 LSystemTag(), LSystemParameters(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo());
}

void PlantArchitect::LSystemBehaviour::PreProcess() {

}

void PlantArchitect::LSystemBehaviour::Grow() {

}

void PlantArchitect::LSystemBehaviour::PostProcess() {
    std::vector<Entity> plants;
    CollectRoots(m_internodesQuery, plants);
    int plantSize = plants.size();

    //Use internal JobSystem to dispatch job for entity collection.
    std::vector<std::shared_future<void>> results;
    for (int plantIndex = 0; plantIndex < plantSize; plantIndex++) {
        results.push_back(JobManager::PrimaryWorkers().Push([&, plantIndex](int id) {
            TreeGraphWalkerEndToRoot(plants[plantIndex], plants[plantIndex], [&](Entity parent){
                float thicknessCollection = 0.0f;
                auto parentInternodeInfo = parent.GetDataComponent<InternodeInfo>();
                auto parameters = parent.GetDataComponent<LSystemParameters>();
                parent.ForEachChild([&](Entity child){
                    if(!InternodeCheck(child)) return;
                    auto childInternodeInfo = child.GetDataComponent<InternodeInfo>();
                    thicknessCollection += glm::pow(childInternodeInfo.m_thickness, 1.0f / parameters.m_thicknessFactor);
                });
                parentInternodeInfo.m_thickness = glm::pow(thicknessCollection, parameters.m_thicknessFactor);
                parent.SetDataComponent(parentInternodeInfo);
            }, [](Entity endNode){
                auto internodeInfo = endNode.GetDataComponent<InternodeInfo>();
                auto parameters = endNode.GetDataComponent<LSystemParameters>();
                internodeInfo.m_thickness = parameters.m_endNodeThickness;
                endNode.SetDataComponent(internodeInfo);
            });
        }).share());
    }
    for (const auto &i: results)
        i.wait();
}

void PlantArchitect::LSystemParameters::OnInspect() {
    ImGui::DragFloat("Internode Length", &m_internodeLength);
    ImGui::DragFloat("Thickness Factor", &m_thicknessFactor);
    ImGui::DragFloat("End node thickness", &m_endNodeThickness);
}
