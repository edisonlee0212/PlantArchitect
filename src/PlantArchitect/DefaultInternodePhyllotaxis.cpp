//
// Created by lllll on 9/6/2021.
//

#include "DefaultInternodePhyllotaxis.hpp"
#include "Internode.hpp"
#include "InternodeLayer.hpp"

using namespace PlantArchitect;

void DefaultInternodePhyllotaxis::GenerateFoliage(const std::shared_ptr<Internode> &internode,
                                                  const InternodeInfo &internodeInfo,
                                                  const GlobalTransform &relativeGlobalTransform, const GlobalTransform &relativeParentGlobalTransform) {
    auto radius = glm::abs(m_positionVariance);
    auto length = internodeInfo.m_length;
    auto rotation = glm::abs(m_randomRotation);
    if(internodeInfo.m_endNode) {
        for (int i = 0; i < m_leafCount; i++) {
            auto position = glm::vec3(glm::gaussRand(0.0f, radius), glm::gaussRand(0.0f, radius), glm::linearRand(0.0f - radius / 2.0f, length + radius / 2.0f));
            const auto transform =
                    relativeGlobalTransform.m_value *
                    (glm::translate(position) *
                     glm::mat4_cast(glm::quat(glm::radians(
                             glm::linearRand(glm::vec3(-rotation), glm::vec3(rotation))))) *
                     glm::scale(glm::vec3(m_leafSize.x, 1.0f, m_leafSize.y)));
            internode->m_foliageMatrices.push_back(transform);
        }
    }
}

void DefaultInternodePhyllotaxis::OnInspect() {
    ImGui::DragInt("Leaf count", &m_leafCount);
    ImGui::DragFloat("Position Variance", &m_positionVariance, 0.01f);
    ImGui::DragFloat("Rotation", &m_randomRotation, 0.01f);
    ImGui::DragFloat2("Leaf size", &m_leafSize.x, 0.01f);
}
void DefaultInternodePhyllotaxis::Serialize(YAML::Emitter &out) {

}

void DefaultInternodePhyllotaxis::Deserialize(const YAML::Node &in) {

}

