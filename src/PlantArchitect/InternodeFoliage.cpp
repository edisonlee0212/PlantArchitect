//
// Created by lllll on 9/6/2021.
//

#include "InternodeFoliage.hpp"
#include "IInternodePhyllotaxis.hpp"
using namespace PlantArchitect;
void InternodeFoliage::Generate(const std::shared_ptr<Internode> &internode,
                                                       const PlantArchitect::InternodeInfo &internodeInfo,
                                                       const GlobalTransform &relativeGlobalTransform, const GlobalTransform &relativeParentGlobalTransform) {
    auto phyllotaxis = m_foliagePhyllotaxis.Get<IInternodePhyllotaxis>();
    if(phyllotaxis){
        phyllotaxis->GenerateFoliage(internode, internodeInfo, relativeGlobalTransform, relativeParentGlobalTransform);
    }
}

void InternodeFoliage::OnInspect() {
    EditorManager::DragAndDropButton(m_foliagePhyllotaxis, "Phyllotaxis", {"EmptyInternodePhyllotaxis", "DefaultInternodePhyllotaxis"}, true);
    EditorManager::DragAndDropButton<Texture2D>(m_foliageTexture, "Texture2D", true);
    if(!m_foliageTexture.Get<Texture2D>()){
        ImGui::DragFloat3("Foliage color", &m_foliageColor.x);
    }
    auto phyllotaxis = m_foliagePhyllotaxis.Get<IInternodePhyllotaxis>();
    if(phyllotaxis){
        if(ImGui::TreeNodeEx("Phyllotaxis", ImGuiTreeNodeFlags_DefaultOpen)) {
            phyllotaxis->OnInspect();
            ImGui::TreePop();
        }
    }
}

void InternodeFoliage::Serialize(YAML::Emitter &out) {
    m_foliagePhyllotaxis.Save("m_foliagePhyllotaxis", out);
    m_foliageTexture.Save("m_foliageTexture", out);
    out << YAML::Key << "m_foliageColor" << YAML::Value << m_foliageColor;
}

void InternodeFoliage::Deserialize(const YAML::Node &in) {
    m_foliagePhyllotaxis.Load("m_foliagePhyllotaxis", in);
    m_foliageTexture.Load("m_foliageTexture", in);
    m_foliageColor = in["m_foliageColor"].as<glm::vec3>();
}

void InternodeFoliage::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_foliagePhyllotaxis);
    list.push_back(m_foliageTexture);
}
