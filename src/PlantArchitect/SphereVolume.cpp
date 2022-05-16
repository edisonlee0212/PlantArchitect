#include <SphereVolume.hpp>
#include "Graphics.hpp"
#include "DefaultResources.hpp"

using namespace PlantArchitect;

void SphereVolume::ApplyMeshRendererBounds() {
    auto meshRenderer =
            GetScene()->GetOrSetPrivateComponent<MeshRenderer>(GetOwner()).lock();
    auto bound = meshRenderer->m_mesh.Get<Mesh>()->GetBound();
    auto difference = bound.m_max - bound.m_min;
    m_radius = (difference.x + difference.y + difference.z) / 6.0f;
}

void SphereVolume::OnCreate() {

    SetEnabled(true);
}

void SphereVolume::OnInspect() {
    ImGui::Checkbox("Obstacle", &m_asObstacle);
    ImGui::DragFloat("Radius", &m_radius, 0.1);
    ImGui::Checkbox("Display bounds", &m_displayBounds);
    if (m_displayBounds) {
        const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
        Graphics::DrawGizmoMesh(
                DefaultResources::Primitives::Sphere, glm::vec4(0, 1, 0, 0.2),
                globalTransform.m_value *
                glm::scale(glm::vec3(m_radius)),
                1);
    }
    if (GetScene()->HasPrivateComponent<MeshRenderer>(GetOwner())) {
        if (ImGui::Button("Apply mesh bound")) {
            ApplyMeshRendererBounds();
        }
    }
}

bool SphereVolume::InVolume(const glm::vec3 &position) {
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    const auto &finalPos = glm::vec3(
            (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
    return glm::length(finalPos) <= m_radius;
}

glm::vec3 SphereVolume::GetRandomPoint() {
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    return glm::vec3((globalTransform.m_value * glm::translate(glm::sphericalRand(m_radius))[3]));
}

bool SphereVolume::InVolume(const GlobalTransform &globalTransform,
                          const glm::vec3 &position) {
    const auto &finalPos = glm::vec3(
            (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
    return glm::length(finalPos) <= m_radius;
}

void SphereVolume::Serialize(YAML::Emitter &out) {
    IVolume::Serialize(out);
    out << YAML::Key << "m_displayPoints" << YAML::Value << m_displayPoints;
    out << YAML::Key << "m_displayBounds" << YAML::Value << m_displayBounds;
    out << YAML::Key << "m_radius" << YAML::Value
        << m_radius;
}

void SphereVolume::Deserialize(const YAML::Node &in) {
    IVolume::Deserialize(in);
    m_displayPoints = in["m_displayPoints"].as<bool>();
    m_displayBounds = in["m_displayBounds"].as<bool>();
    m_radius = in["m_radius"].as<float>();
}

void SphereVolume::OnDestroy() {
    IVolume::OnDestroy();
}
