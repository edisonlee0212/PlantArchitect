//
// Created by lllll on 8/27/2021.
//

#include "Internode.hpp"
#include "InternodeSystem.hpp"
void PlantArchitect::Internode::Clone(const std::shared_ptr<IPrivateComponent> &target) {

}

void PlantArchitect::Internode::OnCreate() {
    m_internodeSystem = EntityManager::GetSystem<InternodeSystem>();
}

void PlantArchitect::Internode::OnRetrieve() {

}

void PlantArchitect::Internode::OnRecycle() {
    m_resource->Reset();
}

void PlantArchitect::Internode::DownStreamResource(float deltaTime) {
    auto owner = GetOwner();
    m_resource->DownStream(deltaTime, owner, owner.GetParent());
}

void PlantArchitect::Internode::UpStreamResource(float deltaTime) {
    auto owner = GetOwner();
    auto children = owner.GetChildren();
    for(const auto& child : children){
        m_resource->UpStream(deltaTime, owner, child);
    }
}

void PlantArchitect::Internode::CollectResource(float deltaTime) {
    m_resource->Collect(deltaTime, GetOwner());
}
