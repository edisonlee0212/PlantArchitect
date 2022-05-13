//
// Created by lllll on 2/26/2022.
//

#include "Root.hpp"
#include "IInternodeFoliage.hpp"

using namespace PlantArchitect;

void Root::OnInspect() {
    ImGui::InputFloat3("Center", &m_center.x, "%.3f", ImGuiInputTextFlags_ReadOnly);
    Editor::DragAndDropButton(m_plantDescriptor, "Descriptor",
                              {"GeneralTreeParameters", "SpaceColonizationParameters", "LSystemString"}, true);
}

void Root::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_center" << YAML::Value << m_center;
    m_plantDescriptor.Save("m_plantDescriptor", out);
}

void Root::Deserialize(const YAML::Node &in) {
    if (in["m_center"]) m_center = in["m_center"].as<glm::vec3>();
    m_plantDescriptor.Load("m_plantDescriptor", in);
}

void Root::OnCreate() {
}

void Root::OnDestroy() {
    m_center = glm::vec3(0.0f);
    m_plantDescriptor.Clear();
}

void Root::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_plantDescriptor);
}

