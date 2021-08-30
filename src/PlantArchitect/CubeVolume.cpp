#include <CubeVolume.hpp>

using namespace PlantArchitect;

void CubeVolume::ApplyMeshRendererBounds() {
    auto meshRenderer =
            GetOwner().GetOrSetPrivateComponent<MeshRenderer>().lock();
    m_minMaxBound = meshRenderer->m_mesh.Get<Mesh>()->GetBound();
}

void CubeVolume::OnCreate() {
    m_minMaxBound.m_min = glm::vec3(-5, 0, -5);
    m_minMaxBound.m_max = glm::vec3(5, 10, 5);
    SetEnabled(true);
}

void CubeVolume::OnGui() {
    ImGui::Checkbox("Obstacle", &m_asObstacle);
    ImGui::DragFloat3("Min", &m_minMaxBound.m_min.x, 0.1);
    ImGui::DragFloat3("Max", &m_minMaxBound.m_max.x, 0.1);
    ImGui::Checkbox("Display bounds", &m_displayBounds);
    if (m_displayBounds) {
        const auto globalTransform = GetOwner().GetDataComponent<GlobalTransform>();
        RenderManager::DrawGizmoMesh(
                DefaultResources::Primitives::Cube, glm::vec4(0, 1, 0, 0.2),
                globalTransform.m_value * glm::translate(m_minMaxBound.Center()) *
                glm::scale(m_minMaxBound.Size()),
                1);
    }
    if (GetOwner().HasPrivateComponent<MeshRenderer>()) {
        if (ImGui::Button("Apply mesh bound")) {
            ApplyMeshRendererBounds();
        }
    }
}

bool CubeVolume::InVolume(const glm::vec3 &position) {
    const auto globalTransform = GetOwner().GetDataComponent<GlobalTransform>();
    const auto &finalPos = glm::vec3(
            (glm::inverse(globalTransform.m_value) * glm::translate(position))[3]);
    return m_minMaxBound.InBound(finalPos);
}

glm::vec3 CubeVolume::GetRandomPoint() {
    const auto globalTransform = GetOwner().GetDataComponent<GlobalTransform>();
    return glm::linearRand(globalTransform.m_value * glm::vec4(m_minMaxBound.m_min, 1.0f), globalTransform.m_value * glm::vec4(m_minMaxBound.m_max, 1.0f));
}

bool CubeVolume::InVolume(const GlobalTransform &globalTransform,
                          const glm::vec3 &position) {
    auto min = glm::vec3(
            (globalTransform.m_value * glm::translate(m_minMaxBound.m_min))[3]);
    auto max = glm::vec3(
            (globalTransform.m_value * glm::translate(m_minMaxBound.m_max))[3]);
    glm::vec3 center = (min + max) / 2.0f;
    glm::vec3 size = (max - min) / 2.0f;

    if (glm::abs(position.x - center.x) > size.x)
        return false;
    if (glm::abs(position.y - center.y) > size.y)
        return false;
    if (glm::abs(position.z - center.z) > size.z)
        return false;
    return true;
}

void CubeVolume::Clone(const std::shared_ptr<IPrivateComponent> &target) {
    *this = *std::static_pointer_cast<CubeVolume>(target);
}

void CubeVolume::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_displayPoints" << YAML::Value << m_displayPoints;
    out << YAML::Key << "m_displayBounds" << YAML::Value << m_displayBounds;
    out << YAML::Key << "m_minMaxBound.m_min" << YAML::Value
        << m_minMaxBound.m_min;
    out << YAML::Key << "m_minMaxBound.m_max" << YAML::Value
        << m_minMaxBound.m_max;
}

void CubeVolume::Deserialize(const YAML::Node &in) {
    m_displayPoints = in["m_displayPoints"].as<bool>();
    m_displayBounds = in["m_displayBounds"].as<bool>();
    m_minMaxBound.m_min = in["m_minMaxBound.m_min"].as<glm::vec3>();
    m_minMaxBound.m_max = in["m_minMaxBound.m_max"].as<glm::vec3>();
}

void CubeVolume::CollectAssetRef(std::vector<AssetRef> &list) {}
