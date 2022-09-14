//
// Created by lllll on 9/11/2022.
//

#include "StrandLayer.hpp"
#include "ClassRegistry.hpp"
#include "Strand.hpp"
using namespace PlantArchitect;

void StrandLayer::OnCreate() {
    ClassRegistry::RegisterPrivateComponent<StrandPlant>("StrandPlant");
}

void StrandLayer::OnInspect() {

}
