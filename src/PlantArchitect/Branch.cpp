//
// Created by lllll on 2/26/2022.
//

#include "Branch.hpp"
using namespace PlantArchitect;


void Branch::OnInspect() {

    ImGui::Text("Chain size: %d", m_internodeChain.size());
    if(ImGui::TreeNode("Chain")){
        for(const auto& i : m_internodeChain){
            ImGui::Text("Index: %d", i.GetIndex());
        }
        ImGui::TreePop();
    }
}

void Branch::Serialize(YAML::Emitter &out) {
    m_currentRoot.Save("m_currentRoot", out);
    m_currentInternode.Save("m_currentInternode", out);
    out << YAML::Key << "m_branchPhysicsParameters" << YAML::Value << YAML::BeginMap;
    out << YAML::EndMap;
}

void Branch::Deserialize(const YAML::Node &in) {
    if("m_currentRoot") m_currentRoot.Load("m_currentRoot", in);
    if("m_currentInternode") m_currentInternode.Load("m_currentInternode", in);
}

void Branch::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_currentRoot.Relink(map);
    m_currentInternode.Relink(map);
}

void Branch::OnDestroy() {
    m_internodeChain.clear();
    m_currentRoot.Clear();
    m_currentInternode.Clear();
}


