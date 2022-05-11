//
// Created by lllll on 2/26/2022.
//

#include "Root.hpp"
#include "IInternodeFoliage.hpp"

using namespace PlantArchitect;

void Root::OnInspect() {
    ImGui::InputFloat3("Center", &m_center.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
}

void Root::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_center" << YAML::Value << m_center;
}

void Root::Deserialize(const YAML::Node &in) {
    if (in["m_center"]) m_center = in["m_center"].as<glm::vec3>();
}

void Root::OnCreate() {
}

void Root::OnDestroy() {
    m_center = glm::vec3(0.0f);
    m_plantDescriptor.Clear();
}

