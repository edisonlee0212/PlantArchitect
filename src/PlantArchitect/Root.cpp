//
// Created by lllll on 2/26/2022.
//

#include "Root.hpp"

void PlantArchitect::Root::OnInspect() {
    ImGui::InputFloat3("Center", &m_center.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
}

void PlantArchitect::Root::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_center" << YAML::Value << m_center;
}

void PlantArchitect::Root::Deserialize(const YAML::Node &in) {
    if (in["m_center"]) m_center = in["m_center"].as<glm::vec3>();
}
