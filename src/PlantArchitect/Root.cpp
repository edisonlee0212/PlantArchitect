//
// Created by lllll on 2/26/2022.
//

#include "Root.hpp"
#include "IInternodePhyllotaxis.hpp"

using namespace PlantArchitect;

void Root::OnInspect() {
    ImGui::InputFloat3("Center", &m_center.x, "%.3f", ImGuiInputTextFlags_ReadOnly);

    Editor::DragAndDropButton(m_foliagePhyllotaxis, "Phyllotaxis",
                              {"EmptyInternodePhyllotaxis", "DefaultInternodePhyllotaxis"}, true);
    Editor::DragAndDropButton<Texture2D>(m_foliageTexture, "Foliage texture", true);
    Editor::DragAndDropButton<Texture2D>(m_branchTexture, "Branch texture", true);
    ImGui::DragFloat3("Foliage color", &m_foliageColor.x);
    ImGui::DragFloat3("Branch color", &m_branchColor.x);

    auto phyllotaxis = m_foliagePhyllotaxis.Get<IInternodePhyllotaxis>();
    if (phyllotaxis) {
        if (ImGui::TreeNodeEx("Phyllotaxis", ImGuiTreeNodeFlags_DefaultOpen)) {
            phyllotaxis->OnInspect();
            ImGui::TreePop();
        }
    }
}

void Root::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_center" << YAML::Value << m_center;

    m_foliagePhyllotaxis.Save("m_foliagePhyllotaxis", out);
    m_foliageTexture.Save("m_foliageTexture", out);
    m_branchTexture.Save("m_branchTexture", out);
    out << YAML::Key << "m_foliageColor" << YAML::Value << m_foliageColor;
    out << YAML::Key << "m_branchColor" << YAML::Value << m_branchColor;
}

void Root::Deserialize(const YAML::Node &in) {
    if (in["m_center"]) m_center = in["m_center"].as<glm::vec3>();

    m_foliagePhyllotaxis.Load("m_foliagePhyllotaxis", in);
    m_foliageTexture.Load("m_foliageTexture", in);
    if (in["m_foliageColor"]) m_foliageColor = in["m_foliageColor"].as<glm::vec3>();

    m_branchTexture.Load("m_branchTexture", in);
    if (in["m_branchColor"]) m_branchColor = in["m_branchColor"].as<glm::vec3>();
}

void Root::OnCreate() {
}

void Root::OnDestroy() {
    m_center = glm::vec3(0.0f);
    m_foliagePhyllotaxis.Clear();
    m_foliageTexture.Clear();
    m_branchTexture.Clear();
    m_plantDescriptor.Clear();
}

void Root::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_foliagePhyllotaxis);
    list.push_back(m_foliageTexture);
}
