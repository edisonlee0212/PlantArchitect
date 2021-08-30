//
// Created by lllll on 8/29/2021.
//

#include "SpaceColonizationBehaviour.hpp"
#include "DefaultInternodeResource.hpp"
using namespace PlantArchitect;

void SpaceColonizationBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Space Colonization Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("Space Colonization Internode", InternodeInfo(),
                                                 SpaceColonizationTag(), SpaceColonizationIncentive(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(), BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeInfo());
}

void SpaceColonizationBehaviour::PreProcess() {
    if (m_attractionPoints.empty()) return;
    std::vector<int> removeMarks;
    removeMarks.resize(m_attractionPoints.size());
    memset(removeMarks.data(), 0, removeMarks.size() * sizeof(bool));
    //1. Check and remove points.
    EntityManager::ForEach<InternodeInfo, GlobalTransform>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag, GlobalTransform &globalTransform) {
                 glm::vec3 position = globalTransform.GetPosition();
                 int index = 0;
                 for (const auto &point: m_attractionPoints) {
                     const float distance2 = point.x * position.x + point.y * position.y + point.z * position.z;
                     if (distance2 < m_removeDistance * m_removeDistance) {
                         removeMarks[index] = 1;
                     }
                     index++;
                 }
             }, true);
    std::vector<glm::vec3> newAttractionPoints;
    for (int i = 0; i < m_attractionPoints.size(); i++) {
        if (removeMarks[i] == 0) {
            newAttractionPoints.push_back(m_attractionPoints[i]);
        }
    }
    std::swap(newAttractionPoints, m_attractionPoints);
    //2. Calculate center
    m_center = glm::vec3(0.0f);
    for (const auto &i: m_attractionPoints) {
        m_center += i;
    }
    m_center /= m_attractionPoints.size();
}

void SpaceColonizationBehaviour::Grow() {
    if (m_attractionPoints.empty()) return;
    //1. Allocate near points
    EntityManager::ForEach<InternodeInfo, GlobalTransform, SpaceColonizationIncentive>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag, GlobalTransform &globalTransform,
                 SpaceColonizationIncentive &spaceColonizationIncentive) {
                 glm::vec3 position = globalTransform.GetPosition();
                 spaceColonizationIncentive.m_direction = glm::vec3(0.0f);
                 spaceColonizationIncentive.m_pointAmount = 0;
                 for (const auto &point: m_attractionPoints) {
                     const float distance2 = point.x * position.x + point.y * position.y + point.z * position.z;
                     if (distance2 < m_attractDistance * m_attractDistance) {
                         spaceColonizationIncentive.m_direction += point - position;
                         spaceColonizationIncentive.m_pointAmount++;
                     }
                 }
             }, true);
    //2. Form new internodes.
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(entities);
    for (const auto &entity: entities) {
        if (!entity.IsEnabled()) continue;
        auto position = entity.GetDataComponent<GlobalTransform>().GetPosition();
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        auto spaceColonizationIncentive = entity.GetDataComponent<SpaceColonizationIncentive>();
        if(spaceColonizationIncentive.m_pointAmount != 0){
            auto newNode = Retrieve<DefaultInternodeResource>(entity);
            GlobalTransform newNodeGlobalTransform;
            newNodeGlobalTransform.m_value = glm::translate(position + m_internodeLength * glm::normalize(spaceColonizationIncentive.m_direction)) * glm::scale(glm::vec3(1.0f));
            newNode.SetDataComponent(newNodeGlobalTransform);
        }else if(internode->m_fromApicalBud){
            auto newNode = Retrieve<DefaultInternodeResource>(entity);
            GlobalTransform newNodeGlobalTransform;
            newNodeGlobalTransform.m_value = glm::translate(position + m_internodeLength * glm::normalize(m_center - position)) * glm::scale(glm::vec3(1.0f));
            newNode.SetDataComponent(newNodeGlobalTransform);
        }
    }
}

void SpaceColonizationBehaviour::PostProcess() {

}

void SpaceColonizationBehaviour::OnInspect() {
    if (ImGui::Button("Create new internode...")) {
        auto entity = Retrieve<DefaultInternodeResource>();
    }
    if (ImGui::Button("Generate branch mesh")) {
        std::vector<Entity> entities;
        m_internodesQuery.ToEntityArray(entities);
        GenerateBranchSkinnedMeshes(entities);
    }

    if(ImGui::TreeNodeEx("Growth", ImGuiTreeNodeFlags_DefaultOpen)){
        static bool renderAttractionPoints = true;
        ImGui::Checkbox("Display attraction points", &renderAttractionPoints);
        if(renderAttractionPoints){

        }
        ImGui::TreePop();
    }
}


