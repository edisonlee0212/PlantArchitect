//
// Created by lllll on 8/16/2021.
//

#include "ObjectRotator.hpp"

void PlantArchitect::ObjectRotator::FixedUpdate() {
    auto scene = GetScene();
    auto transform = scene->GetDataComponent<Transform>(GetOwner());
    m_rotation.y += Application::Time().FixedDeltaTime() * m_rotateSpeed;
    transform.SetEulerRotation(glm::radians(m_rotation));
    scene->SetDataComponent(GetOwner(), transform);
}

void PlantArchitect::ObjectRotator::OnInspect() {
    ImGui::DragFloat("Speed", &m_rotateSpeed);
    ImGui::DragFloat3("Rotation", &m_rotation.x);
}

void PlantArchitect::ObjectRotator::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_rotateSpeed" << YAML::Value << m_rotateSpeed;
    out << YAML::Key << "m_rotation" << YAML::Value << m_rotation;
}

void PlantArchitect::ObjectRotator::Deserialize(const YAML::Node &in) {
    m_rotateSpeed = in["m_rotateSpeed"].as<float>();
    m_rotation = in["m_rotation"].as<glm::vec3>();
}
