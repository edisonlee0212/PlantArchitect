//
// Created by lllll on 8/29/2021.
//

#include "DefaultInternodeBehaviour.hpp"
#include "InternodeSystem.hpp"
#include <DefaultInternodeResource.hpp>

using namespace PlantArchitect;

void DefaultInternodeBehaviour::PreProcess(float deltaTime) {
    EntityManager::ForEach<InternodeInfo, DefaultInternodeTag>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag, DefaultInternodeTag &defaultInternodeTag) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
             }, true);
}

void DefaultInternodeBehaviour::Grow(float deltaTime) {
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(entities);
    EntityManager::ForEach<InternodeInfo, DefaultInternodeTag>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag, DefaultInternodeTag &defaultInternodeTag) {

             }, true);
    for (const auto &entity: entities) {
        if (!entity.IsEnabled()) continue;
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        auto newNode1 = Retrieve(entity);
        auto newNode2 = Retrieve(entity);
        auto newNode3 = Retrieve(entity);
    }
}

void DefaultInternodeBehaviour::PostProcess(float deltaTime) {
    EntityManager::ForEach<InternodeInfo, DefaultInternodeTag>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeInfo &internodeTag, DefaultInternodeTag &defaultInternodeTag) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto children = entity.GetChildren();
                 if (children.size() > 1) Recycle(children[0]);
             }, true);
}

void DefaultInternodeBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Default Internodes");
    }
    m_internodeArchetype =
            EntityManager::CreateEntityArchetype("Default Internode", InternodeInfo(), DefaultInternodeTag(),
                                                 BranchColor(), BranchCylinder(), BranchCylinderWidth(),
                                                 BranchPointer());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(DefaultInternodeTag());
}

void DefaultInternodeBehaviour::OnInspect() {
    RecycleButton();

    if (ImGui::Button("Create new internode...")) {
        auto entity = Retrieve();
    }
    static float resolution = 0.02f;
    static float subdivision = 4.0f;
    ImGui::DragFloat("Resolution", &resolution, 0.001f);
    ImGui::DragFloat("Subdivision", &subdivision, 0.001f);
    if (ImGui::Button("Generate branch mesh")) {
        GenerateSkinnedMeshes(m_internodesQuery, subdivision, resolution);
    }

}

bool DefaultInternodeBehaviour::InternalInternodeCheck(const Entity &target) {
    return target.HasDataComponent<DefaultInternodeTag>();
}

Entity DefaultInternodeBehaviour::Retrieve() {
    return RetrieveHelper<DefaultInternodeResource>();
}

Entity DefaultInternodeBehaviour::Retrieve(const Entity &parent) {
    return RetrieveHelper<DefaultInternodeResource>(parent);
}
