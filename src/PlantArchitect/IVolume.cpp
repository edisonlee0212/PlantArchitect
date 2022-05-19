#include <IVolume.hpp>

using namespace PlantArchitect;

void IVolume::OnInspect() {
    ImGui::Checkbox("Obstacle", &m_asObstacle);
    ImGui::Checkbox("Display bounds", &m_displayBounds);
}

void IVolume::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_asObstacle" << YAML::Value << m_asObstacle;
    out << YAML::Key << "m_displayBounds" << YAML::Value << m_displayBounds;
}

bool IVolume::InVolume(const GlobalTransform &globalTransform, const glm::vec3 &position) { return false; }

bool IVolume::InVolume(const glm::vec3 &position) { return false; }

void IVolume::OnDestroy() {
    m_displayBounds = true;
    m_asObstacle = false;
}

void
IVolume::Deserialize(const YAML::Node &in) {
    if (in["m_displayBounds"]) m_displayBounds = in["m_displayBounds"].as<bool>();
    if (in["m_asObstacle"]) m_asObstacle = in["m_asObstacle"].as<bool>();
}

void IVolume::InVolume(const GlobalTransform &globalTransform, const std::vector<glm::vec3> &positions, std::vector<bool>& results) {
    results.resize(positions.size());
    std::vector<std::shared_future<void>> jobs;
    Jobs::ParallelFor(positions.size(), [&](unsigned i){
        results[i] = InVolume(globalTransform, positions[i]);
    }, jobs);
    for(const auto& i : jobs){
        i.wait();
    }
}

void IVolume::InVolume(const std::vector<glm::vec3> &positions, std::vector<bool>& results) {
    results.resize(positions.size());
    std::vector<std::shared_future<void>> jobs;
    Jobs::ParallelFor(positions.size(), [&](unsigned i){
        results[i] = InVolume(positions[i]);
    }, jobs);
    for(const auto& i : jobs){
        i.wait();
    }
}
