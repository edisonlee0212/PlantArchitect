//
// Created by lllll on 8/27/2021.
//

#include "IInternodeBehaviour.hpp"
#include "Internode.hpp"
using namespace PlantArchitect;
void IInternodeBehaviour::Recycle(const Entity &internode) {
    auto children = internode.GetChildren();
    if (children.empty()) RecycleSingle(internode);
    else {
        for (const auto &child: children) {
            Recycle(child);
        }
    }
}

void IInternodeBehaviour::RecycleSingle(const Entity &internode) {
    std::lock_guard<std::mutex> lockGuard(m_internodeFactoryLock);
    internode.GetOrSetPrivateComponent<Internode>().lock()->OnRecycle();
    internode.SetParent(m_recycleStorageEntity.Get());
    internode.SetEnabled(false);
    m_recycledInternodes.emplace_back(internode);
}

void IInternodeBehaviour::GenerateBranchSkinnedMeshes(const std::vector<Entity> &entities) {
    for(const auto& entity : entities){

    }
}