#include <CylinderVolume.hpp>
#include "Graphics.hpp"
#include "DefaultResources.hpp"

using namespace PlantArchitect;

void CylinderVolume::ApplyMeshRendererBounds() {
    auto meshRenderer =
            GetScene()->GetOrSetPrivateComponent<MeshRenderer>(GetOwner()).lock();
    auto bound = meshRenderer->m_mesh.Get<Mesh>()->GetBound();
    auto difference = bound.m_max - bound.m_min;
    m_radius = m_height = (difference.x + difference.y + difference.z) / 6.0f;
}

void CylinderVolume::OnCreate() {
    SetEnabled(true);
}

void CylinderVolume::OnInspect() {
    ImGui::Checkbox("Obstacle", &m_asObstacle);
    ImGui::DragFloat("Radius", &m_radius, 0.1);
    ImGui::DragFloat("Height", &m_height, 0.1);
    ImGui::Checkbox("Display bounds", &m_displayBounds);
    if (m_displayBounds) {
        const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
        Graphics::DrawGizmoMesh(
                DefaultResources::Primitives::Cylinder, glm::vec4(0, 1, 0, 0.2),
                globalTransform.m_value *
                glm::scale(glm::vec3(m_radius, m_height, m_radius)),
                1);
    }
    if (GetScene()->HasPrivateComponent<MeshRenderer>(GetOwner())) {
        if (ImGui::Button("Apply mesh bound")) {
            ApplyMeshRendererBounds();
        }
    }
}

bool CylinderVolume::InVolume(const glm::vec3 &position) {
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    const auto &finalPos = glm::vec3(
            (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
    return (finalPos.x - m_radius) * (finalPos.x - m_radius) + (finalPos.y - m_radius) * (finalPos.y - m_radius) <=
           m_radius * m_radius && (finalPos.y - m_height) * (finalPos.y - m_height) <= (m_height / 2.0f) * (m_height / 2.0f);
}

glm::vec3 CylinderVolume::GetRandomPoint() {
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    return glm::vec3((globalTransform.m_value * glm::translate(glm::vec3(glm::diskRand(m_radius), glm::linearRand(-m_height, m_height)))[3]));
}

bool CylinderVolume::InVolume(const GlobalTransform &globalTransform,
                              const glm::vec3 &position) {
    const auto &finalPos = glm::vec3(
            (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
    return glm::length(finalPos) <= m_radius;
}

void CylinderVolume::Serialize(YAML::Emitter &out) {
    IVolume::Serialize(out);
    out << YAML::Key << "m_displayPoints" << YAML::Value << m_displayPoints;
    out << YAML::Key << "m_displayBounds" << YAML::Value << m_displayBounds;
    out << YAML::Key << "m_radius" << YAML::Value << m_radius;
    out << YAML::Key << "m_height" << YAML::Value << m_height;
}

void CylinderVolume::Deserialize(const YAML::Node &in) {
    IVolume::Deserialize(in);
    m_displayPoints = in["m_displayPoints"].as<bool>();
    m_displayBounds = in["m_displayBounds"].as<bool>();
    m_radius = in["m_radius"].as<float>();
    m_height = in["m_height"].as<float>();
}

void CylinderVolume::OnDestroy() {
    IVolume::OnDestroy();
}
