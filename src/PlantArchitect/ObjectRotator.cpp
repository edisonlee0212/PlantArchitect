//
// Created by lllll on 8/16/2021.
//

#include "ObjectRotator.hpp"
void PlantArchitect::ObjectRotator::FixedUpdate() {
  auto transform = GetOwner().GetDataComponent<Transform>();
  m_rotation.y += Application::Time().FixedDeltaTime() * m_rotateSpeed;
  transform.SetEulerRotation(glm::radians(m_rotation));
  GetOwner().SetDataComponent(transform);
}
void PlantArchitect::ObjectRotator::Clone(
    const std::shared_ptr<IPrivateComponent> &target) {
  *this = *std::static_pointer_cast<ObjectRotator>(target);
}
void PlantArchitect::ObjectRotator::OnGui() {
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
