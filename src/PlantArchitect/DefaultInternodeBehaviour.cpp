//
// Created by lllll on 8/29/2021.
//

#include "DefaultInternodeBehaviour.hpp"
#include "InternodeSystem.hpp"

void PlantArchitect::DefaultInternodeBehaviour::Grow() {

}

void PlantArchitect::DefaultInternodeBehaviour::PostProcess() {

}

void PlantArchitect::DefaultInternodeBehaviour::PreProcess() {

}

void PlantArchitect::DefaultInternodeBehaviour::OnCreate() {
    if (m_recycleStorageEntity.Get().IsNull()) {
        m_recycleStorageEntity = EntityManager::CreateEntity("Recycled Default Internodes");
    }
    m_internodeArchetype = EntityManager::CreateEntityArchetype("Default Internode", InternodeTag(), DefaultInternodeTag());
    m_internodesQuery = EntityManager::CreateEntityQuery();
    m_internodesQuery.SetAllFilters(InternodeTag());
}

void PlantArchitect::DefaultInternodeBehaviour::OnInspect() {
    if(ImGui::Button("Create new internode...")){

    }
}
