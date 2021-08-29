//
// Created by lllll on 8/29/2021.
//

#include "DefaultInternodeBehaviour.hpp"
#include "InternodeSystem.hpp"
#include <DefaultInternodeResource.hpp>
using namespace PlantArchitect;

void DefaultInternodeBehaviour::PreProcess() {
    EntityManager::ForEach<InternodeTag, DefaultInternodeTag>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeTag &internodeTag, DefaultInternodeTag &defaultInternodeTag) {
                auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
             }, true);
}

void DefaultInternodeBehaviour::Grow() {
    std::vector<Entity> entities;
    m_internodesQuery.ToEntityArray(entities);
    EntityManager::ForEach<InternodeTag, DefaultInternodeTag>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeTag &internodeTag, DefaultInternodeTag &defaultInternodeTag) {

             }, true);
    for(const auto& entity : entities){
        if(!entity.IsEnabled()) continue;
        auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
        auto newNode1 = Retrieve<DefaultInternodeResource>(entity);
        auto newNode2 = Retrieve<DefaultInternodeResource>(entity);
        auto newNode3 = Retrieve<DefaultInternodeResource>(entity);
    }
}

void DefaultInternodeBehaviour::PostProcess() {
    EntityManager::ForEach<InternodeTag, DefaultInternodeTag>
            (JobManager::PrimaryWorkers(), m_internodesQuery,
             [&](int i, Entity entity, InternodeTag &internodeTag, DefaultInternodeTag &defaultInternodeTag) {
                 auto internode = entity.GetOrSetPrivateComponent<Internode>().lock();
                 auto children = entity.GetChildren();
                 if(children.size() > 1) Recycle(children[0]);

             }, true);
}

void DefaultInternodeBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Default Internodes");
    }
    m_internodeArchetype = EntityManager::CreateEntityArchetype("Default Internode", InternodeTag(),
                                                                DefaultInternodeTag());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeTag());
}

void DefaultInternodeBehaviour::OnInspect() {
    if (ImGui::Button("Create new internode...")) {
        auto entity = Retrieve<DefaultInternodeResource>();
    }
}
