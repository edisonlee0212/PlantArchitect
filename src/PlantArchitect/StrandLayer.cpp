//
// Created by lllll on 9/11/2022.
//

#include "StrandLayer.hpp"
#include "ClassRegistry.hpp"
#include "Strand.hpp"
#include "DataComponents.hpp"
using namespace PlantArchitect;

void StrandLayer::OnCreate() {
    ClassRegistry::RegisterPrivateComponent<StrandPlant>("StrandPlant");
    ClassRegistry::RegisterPrivateComponent<StrandsIntersection>("StrandsIntersection");
    ClassRegistry::RegisterDataComponent<StrandIntersectionInfo>("StrandIntersectionInfo");

    m_strandIntersectionArchetype =
            Entities::CreateEntityArchetype("Strand Intersection", StrandIntersectionInfo());
}

void StrandLayer::OnInspect() {

}
