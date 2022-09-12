//
// Created by lllll on 3/25/2022.
//

#include "InternodeModel/IPlantDescriptor.hpp"
#include "IInternodeFoliage.hpp"
using namespace PlantArchitect;
void IPlantDescriptor::OnInspect() {
    if(ImGui::Button("Instantiate")){
        InstantiateTree();
    }
    ImGui::DragFloat3("Foliage color", &m_foliageColor.x);
    ImGui::DragFloat3("Branch color", &m_branchColor.x);
    Editor::DragAndDropButton(m_foliagePhyllotaxis, "Phyllotaxis",
                              {"DefaultInternodeFoliage"}, true);
    Editor::DragAndDropButton<Texture2D>(m_branchTexture, "Branch texture", true);
}

void IPlantDescriptor::Serialize(YAML::Emitter &out) {
    m_foliagePhyllotaxis.Save("m_foliagePhyllotaxis", out);
    m_branchTexture.Save("m_branchTexture", out);
    out << YAML::Key << "m_foliageColor" << YAML::Value << m_foliageColor;
    out << YAML::Key << "m_branchColor" << YAML::Value << m_branchColor;
}

void IPlantDescriptor::Deserialize(const YAML::Node &in) {
    m_foliagePhyllotaxis.Load("m_foliagePhyllotaxis", in);
    m_branchTexture.Load("m_branchTexture", in);
    if (in["m_foliageColor"]) m_foliageColor = in["m_foliageColor"].as<glm::vec3>();
    if (in["m_branchColor"]) m_branchColor = in["m_branchColor"].as<glm::vec3>();
}

void IPlantDescriptor::CollectAssetRef(std::vector<AssetRef> &list) {
    list.emplace_back(m_branchTexture);
    list.emplace_back(m_foliagePhyllotaxis);
}
