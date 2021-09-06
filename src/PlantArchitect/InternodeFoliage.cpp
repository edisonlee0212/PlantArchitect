//
// Created by lllll on 9/6/2021.
//

#include "InternodeFoliage.hpp"
#include "IInternodePhyllotaxis.hpp"

using namespace PlantArchitect;
void InternodeFoliage::Generate(const std::shared_ptr<Internode> &internode,
                                                       const PlantArchitect::InternodeInfo &internodeInfo,
                                                       const GlobalTransform &relativeGlobalTransform) {
    auto phyllotaxis = m_foliagePhyllotaxis.Get<IInternodePhyllotaxis>();
    if(phyllotaxis){
        phyllotaxis->GenerateFoliage(internode, internodeInfo, relativeGlobalTransform);
    }
}

void InternodeFoliage::OnInspect() {
    DragAndDropButton<IInternodePhyllotaxis>(m_foliagePhyllotaxis, "Phyllotaxis", {"EmptyInternodePhyllotaxis", "DefaultInternodePhyllotaxis"}, true);
}

