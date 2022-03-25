//
// Created by lllll on 3/25/2022.
//

#include "IPlantDescriptor.hpp"
using namespace PlantArchitect;
void IPlantDescriptor::OnInspect() {
    if(ImGui::Button("Instantiate")){
        InstantiateTree();
    }
}
