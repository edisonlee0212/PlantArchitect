//
// Created by lllll on 9/6/2021.
//

#include "DefaultInternodeFoliage.hpp"
#include "InternodeModel/Internode.hpp"
#include "InternodeLayer.hpp"

using namespace PlantArchitect;

void DefaultInternodeFoliage::GenerateFoliage(const std::shared_ptr<Internode> &internode,
                                              const InternodeInfo &internodeInfo,
                                              const GlobalTransform &relativeGlobalTransform,
                                              const GlobalTransform &relativeParentGlobalTransform) {
    auto radius = glm::abs(m_positionVariance);
    auto length = internodeInfo.m_length;
    auto rotation = glm::abs(m_randomRotation);
    if (m_endNodeOnly && !internodeInfo.m_endNode) return;
    if (internodeInfo.m_rootDistance < glm::min(m_rootDistanceRange.x, m_rootDistanceRange.y) ||
        internodeInfo.m_rootDistance > glm::max(m_rootDistanceRange.x, m_rootDistanceRange.y))
        return;
    if (internodeInfo.m_maxDistanceToAnyBranchEnd <
        glm::min(m_distanceToBranchEndRange.x, m_distanceToBranchEndRange.y) ||
        internodeInfo.m_maxDistanceToAnyBranchEnd >
        glm::max(m_distanceToBranchEndRange.x, m_distanceToBranchEndRange.y))
        return;
    auto position = relativeGlobalTransform.GetPosition();
    if (position.y < glm::min(m_heightRange.x, m_heightRange.y) ||
        position.y > glm::max(m_heightRange.x, m_heightRange.y))
        return;
    for (int i = 0; i < m_leafCount; i++) {
        auto foliagePosition = glm::vec3(glm::gaussRand(0.0f, radius), glm::gaussRand(0.0f, radius),
                                         -glm::linearRand(0.0f - radius / 2.0f, length + radius / 2.0f));
        const auto transform =
                relativeGlobalTransform.m_value *
                (glm::translate(foliagePosition) *
                 glm::mat4_cast(glm::quat(glm::radians(
                         glm::linearRand(glm::vec3(-rotation), glm::vec3(rotation))))) *
                 glm::scale(glm::vec3(m_leafSize.x, 1.0f, m_leafSize.y)));
        internode->m_foliageMatrices.push_back(transform);
    }
}

void DefaultInternodeFoliage::OnInspect() {
    IInternodeFoliage::OnInspect();
    ImGui::Checkbox("End node only", &m_endNodeOnly);
    ImGui::DragFloat2("Height limit", &m_heightRange.x, 0.01f);
    ImGui::DragFloat2("Root distance", &m_rootDistanceRange.x, 0.01f);
    ImGui::DragFloat2("End distance", &m_distanceToBranchEndRange.x, 0.01f);

    ImGui::DragInt("Leaf count", &m_leafCount);
    ImGui::DragFloat("Position Variance", &m_positionVariance, 0.01f);
    ImGui::DragFloat("Rotation", &m_randomRotation, 0.01f);
    ImGui::DragFloat2("Leaf size", &m_leafSize.x, 0.01f);
}

void DefaultInternodeFoliage::Serialize(YAML::Emitter &out) {
    IInternodeFoliage::Serialize(out);

    out << YAML::Key << "m_endNodeOnly" << YAML::Value << m_endNodeOnly;
    out << YAML::Key << "m_heightRange" << YAML::Value << m_heightRange;
    out << YAML::Key << "m_rootDistanceRange" << YAML::Value << m_rootDistanceRange;
    out << YAML::Key << "m_distanceToBranchEndRange" << YAML::Value << m_distanceToBranchEndRange;

    out << YAML::Key << "m_positionVariance" << YAML::Value << m_positionVariance;
    out << YAML::Key << "m_randomRotation" << YAML::Value << m_randomRotation;
    out << YAML::Key << "m_leafSize" << YAML::Value << m_leafSize;
    out << YAML::Key << "m_leafCount" << YAML::Value << m_leafCount;
}

void DefaultInternodeFoliage::Deserialize(const YAML::Node &in) {
    IInternodeFoliage::Deserialize(in);

    if (in["m_endNodeOnly"]) m_endNodeOnly = in["m_endNodeOnly"].as<bool>();
    if (in["m_heightRange"]) m_heightRange = in["m_heightRange"].as<glm::vec2>();
    if (in["m_rootDistanceRange"]) m_rootDistanceRange = in["m_rootDistanceRange"].as<glm::vec2>();
    if (in["m_distanceToBranchEndRange"]) m_distanceToBranchEndRange = in["m_distanceToBranchEndRange"].as<glm::vec2>();

    if (in["m_positionVariance"]) m_positionVariance = in["m_positionVariance"].as<float>();
    if (in["m_randomRotation"]) m_randomRotation = in["m_randomRotation"].as<float>();
    if (in["m_leafSize"]) m_leafSize = in["m_leafSize"].as<glm::vec2>();
    if (in["m_leafCount"]) m_leafCount = in["m_leafCount"].as<int>();
}

void DefaultInternodeFoliage::CollectAssetRef(std::vector<AssetRef> &list) {
    IInternodeFoliage::CollectAssetRef(list);
}

