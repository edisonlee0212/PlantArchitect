//
// Created by lllll on 9/11/2021.
//

#include "GeneralTreeParameters.hpp"

using namespace PlantArchitect;

void GeneralTreeParameters::OnInspect() {
    if(ImGui::TreeNodeEx("Structure", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragInt("Lateral bud per node", &m_lateralBudCount);
        ImGui::DragFloat2("Branching Angle mean/var", &m_branchingAngleMeanVariance.x, 0.01f);
        ImGui::DragFloat2("Roll Angle mean/var", &m_rollAngleMeanVariance.x, 0.01f);
        ImGui::DragFloat2("Apical Angle mean/var", &m_apicalAngleMeanVariance.x, 0.01f);
        ImGui::DragFloat("Gravitropism", &m_gravitropism, 0.01f);
        ImGui::DragFloat("Phototropism", &m_phototropism, 0.01f);
        ImGui::DragFloat2("Internode length mean/var", &m_internodeLengthMeanVariance.x, 0.01f);
        ImGui::DragFloat2("Thickness min/factor", &m_endNodeThicknessAndControl.x, 0.01f);
        ImGui::TreePop();
    }
    if(ImGui::TreeNodeEx("Bud", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat("Lateral bud flushing probability", &m_lateralBudFlushingProbability, 0.01f);
        ImGui::DragFloat3("Neighbor avoidance mul/factor/max", &m_neighborAvoidance.x, 0.001f);
        ImGui::DragFloat("Apical control", &m_apicalControl, 0.01f);
        ImGui::DragFloat3("Apical dominance base/age/dist", &m_apicalDominanceBaseAgeDist.x, 0.01f);
        int maxAgeBeforeInhibitorEnds = m_apicalDominanceBaseAgeDist.x / m_apicalDominanceBaseAgeDist.y;
        float maxDistance = m_apicalDominanceBaseAgeDist.x / m_apicalDominanceBaseAgeDist.z;
        ImGui::Text("Max age / distance: [%i, %.3f]", maxAgeBeforeInhibitorEnds, maxDistance);

        ImGui::DragFloat("Lateral bud lighting factor", &m_lateralBudFlushingLightingFactor, 0.01f);
        ImGui::DragFloat2("Kill probability apical/lateral", &m_budKillProbabilityApicalLateral.x, 0.01f);
        ImGui::TreePop();
    }
    if(ImGui::TreeNodeEx("Internode")) {
        ImGui::DragInt("Random pruning Order Protection", &m_randomPruningOrderProtection);
        ImGui::DragFloat3("Random pruning base/age/max", &m_randomPruningBaseAgeMax.x, 0.0001f, -1.0f, 1.0f, "%.5f");
        const float maxAgeBeforeMaxCutOff =
                (m_randomPruningBaseAgeMax.z - m_randomPruningBaseAgeMax.x) / m_randomPruningBaseAgeMax.y;
        ImGui::Text("Max age before reaching max: %.3f", maxAgeBeforeMaxCutOff);
        ImGui::DragFloat("Low Branch Pruning", &m_lowBranchPruning, 0.01f);
        ImGui::DragFloat3("Sagging thickness/reduction/max", &m_saggingFactorThicknessReductionMax.x, 0.01f);
        ImGui::TreePop();
    }
    ImGui::DragInt("Mature age", &m_matureAge, 1, 0, 1000);


}

GeneralTreeParameters::GeneralTreeParameters() {
    m_lateralBudCount = 2;
    m_branchingAngleMeanVariance = glm::vec2(30, 3);
    m_rollAngleMeanVariance = glm::vec2(120, 2);
    m_apicalAngleMeanVariance = glm::vec2(20, 2);
    m_gravitropism = -0.1f;
    m_phototropism = 0.05f;
    m_internodeLengthMeanVariance = glm::vec2(1, 0.1);
    m_endNodeThicknessAndControl = glm::vec2(0.01, 0.5);
    m_lateralBudFlushingProbability = 0.3f;
    m_neighborAvoidance = glm::vec3(0.05f, 1, 100);
    m_apicalControl = 2.0f;
    m_apicalDominanceBaseAgeDist = glm::vec3(0.12, 1, 0.3);
    m_lateralBudFlushingLightingFactor = 0.0f;
    m_budKillProbabilityApicalLateral = glm::vec2(0.0, 0.03);
    m_randomPruningOrderProtection = 1;
    m_randomPruningBaseAgeMax = glm::vec3(-0.1, 0.007, 0.5);
    m_lowBranchPruning = 0.15f;
    m_saggingFactorThicknessReductionMax = glm::vec3(6, 3, 0.5);
    m_matureAge = 30;
}

void GeneralTreeParameters::Save(const std::filesystem::path &path) const {
    auto directory = path;
    directory.remove_filename();
    std::filesystem::create_directories(directory);
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "m_lateralBudCount" << YAML::Value << m_lateralBudCount;
    out << YAML::Key << "m_branchingAngleMeanVariance" << YAML::Value << m_branchingAngleMeanVariance;
    out << YAML::Key << "m_rollAngleMeanVariance" << YAML::Value << m_rollAngleMeanVariance;
    out << YAML::Key << "m_apicalAngleMeanVariance" << YAML::Value << m_apicalAngleMeanVariance;
    out << YAML::Key << "m_gravitropism" << YAML::Value << m_gravitropism;
    out << YAML::Key << "m_phototropism" << YAML::Value << m_phototropism;
    out << YAML::Key << "m_internodeLengthMeanVariance" << YAML::Value << m_internodeLengthMeanVariance;
    out << YAML::Key << "m_endNodeThicknessAndControl" << YAML::Value << m_endNodeThicknessAndControl;
    out << YAML::Key << "m_lateralBudFlushingProbability" << YAML::Value << m_lateralBudFlushingProbability;
    out << YAML::Key << "m_neighborAvoidance" << YAML::Value << m_neighborAvoidance;
    out << YAML::Key << "m_apicalControl" << YAML::Value << m_apicalControl;
    out << YAML::Key << "m_apicalDominanceBaseAgeDist" << YAML::Value << m_apicalDominanceBaseAgeDist;
    out << YAML::Key << "m_lateralBudFlushingLightingFactor" << YAML::Value << m_lateralBudFlushingLightingFactor;
    out << YAML::Key << "m_budKillProbabilityApicalLateral" << YAML::Value << m_budKillProbabilityApicalLateral;
    out << YAML::Key << "m_randomPruningOrderProtection" << YAML::Value << m_randomPruningOrderProtection;
    out << YAML::Key << "m_randomPruningBaseAgeMax" << YAML::Value << m_randomPruningBaseAgeMax;
    out << YAML::Key << "m_lowBranchPruning" << YAML::Value << m_lowBranchPruning;
    out << YAML::Key << "m_saggingFactorThicknessReductionMax" << YAML::Value << m_saggingFactorThicknessReductionMax;
    out << YAML::Key << "m_matureAge" << YAML::Value << m_matureAge;


    out << YAML::EndMap;
    std::ofstream fout(path.string());
    fout << out.c_str();
    fout.flush();
}

void GeneralTreeParameters::Load(const std::filesystem::path &path) {
    std::ifstream stream(path.string());
    std::stringstream stringStream;
    stringStream << stream.rdbuf();
    YAML::Node in = YAML::Load(stringStream.str());
    if(in["m_lateralBudCount"]) m_lateralBudCount = in["m_lateralBudCount"].as<int>();
    if(in["m_branchingAngleMeanVariance"]) m_branchingAngleMeanVariance = in["m_branchingAngleMeanVariance"].as<glm::vec2>();
    if(in["m_rollAngleMeanVariance"]) m_rollAngleMeanVariance = in["m_rollAngleMeanVariance"].as<glm::vec2>();
    if(in["m_apicalAngleMeanVariance"]) m_apicalAngleMeanVariance = in["m_apicalAngleMeanVariance"].as<glm::vec2>();
    if(in["m_gravitropism"]) m_gravitropism = in["m_gravitropism"].as<float>();
    if(in["m_phototropism"]) m_phototropism = in["m_phototropism"].as<float>();
    if(in["m_internodeLengthMeanVariance"]) m_internodeLengthMeanVariance = in["m_internodeLengthMeanVariance"].as<glm::vec2>();
    if(in["m_endNodeThicknessAndControl"]) m_endNodeThicknessAndControl = in["m_endNodeThicknessAndControl"].as<glm::vec2>();
    if(in["m_lateralBudFlushingProbability"]) m_lateralBudFlushingProbability = in["m_lateralBudFlushingProbability"].as<float>();
    if(in["m_neighborAvoidance"]) m_neighborAvoidance = in["m_neighborAvoidance"].as<glm::vec3>();
    if(in["m_apicalControl"]) m_apicalControl = in["m_apicalControl"].as<float>();
    if(in["m_apicalDominanceBaseAgeDist"]) m_apicalDominanceBaseAgeDist = in["m_apicalDominanceBaseAgeDist"].as<glm::vec3>();
    if(in["m_lateralBudFlushingLightingFactor"]) m_lateralBudFlushingLightingFactor = in["m_lateralBudFlushingLightingFactor"].as<float>();
    if(in["m_budKillProbabilityApicalLateral"]) m_budKillProbabilityApicalLateral = in["m_budKillProbabilityApicalLateral"].as<glm::vec2>();
    if(in["m_randomPruningOrderProtection"])  m_randomPruningOrderProtection = in["m_randomPruningOrderProtection"].as<int>();
    if(in["m_randomPruningBaseAgeMax"]) m_randomPruningBaseAgeMax = in["m_randomPruningBaseAgeMax"].as<glm::vec3>();
    if(in["m_lowBranchPruning"]) m_lowBranchPruning = in["m_lowBranchPruning"].as<float>();
    if(in["m_saggingFactorThicknessReductionMax"]) m_saggingFactorThicknessReductionMax = in["m_saggingFactorThicknessReductionMax"].as<glm::vec3>();
    if(in["m_matureAge"]) m_matureAge = in["m_matureAge"].as<int>();


}

void InternodeStatus::OnInspect() {
    ImGui::Text(("Branching Order: " + std::to_string(m_branchingOrder)).c_str());
    ImGui::Text(("Age: " + std::to_string(m_age)).c_str());
    ImGui::Text(("Sagging: " + std::to_string(m_sagging)).c_str());
    ImGui::Text(("Inhibitor: " + std::to_string(m_inhibitor)).c_str());
    ImGui::Text(("DistanceToRoot: " + std::to_string(m_distanceToRoot)).c_str());
    ImGui::Text(("MaxDistanceToAnyBranchEnd: " + std::to_string(m_maxDistanceToAnyBranchEnd)).c_str());
    ImGui::Text(("TotalDistanceToAllBranchEnds: " + std::to_string(m_totalDistanceToAllBranchEnds)).c_str());
    ImGui::Text(("Order: " + std::to_string(m_order)).c_str());
    ImGui::Text(("Level: " + std::to_string(m_level)).c_str());
    ImGui::Text(("Biomass: " + std::to_string(m_biomass)).c_str());
    ImGui::Text(("ChildTotalBiomass: " + std::to_string(m_childTotalBiomass)).c_str());
    glm::vec3 localRotation = glm::eulerAngles(m_desiredLocalRotation);
    ImGui::Text(("Desired local rotation: [" + std::to_string(glm::degrees(localRotation.x)) + ", " + std::to_string(glm::degrees(localRotation.y)) + ", " +std::to_string(glm::degrees(localRotation.z)) + "]").c_str());

}

void InternodeStatus::CalculateApicalControl(float apicalControl) {
    m_apicalControl = glm::pow(1.0f / apicalControl, m_level);
}

void InternodeWaterPressure::OnInspect() {
    ImGui::Text(("m_value: " + std::to_string(m_value)).c_str());
}

void InternodeWater::OnInspect() {
    ImGui::Text(("m_value: " + std::to_string(m_value)).c_str());
}

void InternodeIllumination::OnInspect() {
    ImGui::Text(("Intensity: " + std::to_string(m_intensity)).c_str());
    ImGui::Text(("Direction: [" + std::to_string(glm::degrees(m_direction.x)) + ", " +
                 std::to_string(glm::degrees(m_direction.y)) + ", " + std::to_string(glm::degrees(m_direction.z)) +
                 "]").c_str());
}