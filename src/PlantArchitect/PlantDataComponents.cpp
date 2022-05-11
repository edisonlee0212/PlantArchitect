//
// Created by lllll on 10/26/2021.
//

#include "PlantDataComponents.hpp"
using namespace PlantArchitect;

void InternodeInfo::OnInspect() {
    ImGui::Text(("DistanceToRoot: " + std::to_string(m_rootDistance)).c_str());
    ImGui::Text(("MaxDistanceToAnyBranchEnd: " + std::to_string(m_maxDistanceToAnyBranchEnd)).c_str());
    ImGui::Text(("TotalDistanceToAllBranchEnds: " + std::to_string(m_totalDistanceToAllBranchEnds)).c_str());
    ImGui::Text(("Order: " + std::to_string(m_order)).c_str());
    ImGui::Text(("Biomass: " + std::to_string(m_biomass)).c_str());
    ImGui::Text(("ChildTotalBiomass: " + std::to_string(m_childTotalBiomass)).c_str());

    ImGui::Text(("Proximity: " + std::to_string(m_neighborsProximity)).c_str());

    ImGui::Text(("Thickness: " + std::to_string(m_thickness)).c_str());
    ImGui::Text(("Length: " + std::to_string(m_length)).c_str());
    ImGui::Text(("Is end node: " + std::to_string(m_endNode)).c_str());

    glm::vec3 localRotation = glm::eulerAngles(m_localRotation);
    ImGui::Text(("Local Rotation: [" + std::to_string(glm::degrees(localRotation.x)) + ", " + std::to_string(glm::degrees(localRotation.y)) + ", " +std::to_string(glm::degrees(localRotation.z)) + "]").c_str());
}
void InternodeStatistics::OnInspect() {
    ImGui::Text(("L-System: " + std::to_string(m_lSystemStringIndex)).c_str());
    ImGui::Text(("Strahler: " + std::to_string(m_strahlerOrder)).c_str());
    ImGui::Text(("Horton: " + std::to_string(m_hortonOrdering)).c_str());
}

void InternodeColor::OnInspect() {
    ImGui::ColorEdit4("Color", &m_value.x);
}

void InternodeCylinderWidth::OnInspect() {
    ImGui::Text(("Value: " + std::to_string(m_value)).c_str());
}

void BranchColor::OnInspect() {
    ImGui::ColorEdit4("Color", &m_value.x);
}

void BranchCylinderWidth::OnInspect() {
    ImGui::Text(("Value: " + std::to_string(m_value)).c_str());
}

void BranchInfo::OnInspect() {
    ImGui::Text(("Thickness: " + std::to_string(m_thickness)).c_str());
    ImGui::Text(("Length: " + std::to_string(m_length)).c_str());
    ImGui::Text(("Is end node: " + std::to_string(m_endNode)).c_str());
}
