//
// Created by lllll on 12/28/2021.
//

#include "FBMField.hpp"

void PlantArchitect::FBMField::OnCreate() {
}

void PlantArchitect::FBMField::Serialize(YAML::Emitter &out) {
}

void PlantArchitect::FBMField::Deserialize(const YAML::Node &in) {
}

void PlantArchitect::FBMField::OnInspect() {
    static bool draw = true;
    ImGui::Checkbox("Render field", &draw);
    if (draw) {
        static auto minRange = glm::vec3(-5, 0, -5);
        static auto maxRange = glm::vec3(5, 0, 5);
        static float step = 0.2f;
        if (ImGui::DragFloat3("Min", &minRange.x, 0.1f)) {
            minRange = (glm::min) (minRange, maxRange);
        }
        if (ImGui::DragFloat3("Max", &maxRange.x, 0.1f)) {
            maxRange = (glm::max) (minRange, maxRange);
        }
        ImGui::DragFloat("Step", &step, 0.01f);
        step = glm::clamp(step, 0.1f, 10.0f);
        static std::vector<glm::vec4> colors;
        static std::vector<glm::mat4> matrices;
        const int sx = (int) ((maxRange.x - minRange.x + step) / step);
        const int sy = (int) ((maxRange.y - minRange.y + step) / step);
        const int sz = (int) ((maxRange.z - minRange.z + step) / step);
        colors.resize(sx * sy * sz);
        matrices.resize(sx * sy * sz);
        int i = 0;
        static float width = 0.1f;
        ImGui::DragFloat("Width", &width, 0.01f);

        static glm::vec3 frequency = {2.01f, 3.03f, 1.04f};
        static glm::vec3 density = {0.167f, 0.297f, 0.123f};
        static glm::vec3 factor = {0.33f, 0.33f, 0.33f};
        static float t = 0;
        static bool pushTime = true;
        ImGui::Checkbox("Time", &pushTime);
        if (pushTime) {
            t += Application::Time().DeltaTime();
        }
        ImGui::DragFloat3("Freq", &frequency.x, 0.01f);
        ImGui::DragFloat3("Density", &density.x, 0.001f);
        ImGui::DragFloat3("Factor", &factor.x, 0.01f);
        ImGui::DragFloat("t", &t);
        for (float x = minRange.x; x <= maxRange.x; x += step) {
            for (float y = minRange.y; y <= maxRange.y; y += step) {
                for (float z = minRange.z; z <= maxRange.z; z += step) {
                    glm::vec3 coord = {x, y, z};
                    colors[i] = {factor.x * GetT(coord, t, frequency.x, density.x, 6) +
                                 factor.y * GetT(coord, t, frequency.y, density.y, 6) +
                                 factor.z * GetT(coord, t, frequency.z, density.z, 6), 1.0f};
                    matrices[i] = glm::translate(glm::vec3(x, y, z)) * glm::scale(glm::vec3(width));
                    i++;
                }
            }
        }

        RenderManager::DrawGizmoMeshInstancedColored(DefaultResources::Primitives::Cube, colors, matrices);
    }
}

float PlantArchitect::FBMField::Get(const glm::vec3 &in, unsigned int level) {
    float f = 0;
    float sum = 0.0f;
    float factor = 0.5f;
    glm::vec3 p = in;
    for (unsigned i = 0; i < level; i++) {
        f += factor * m_noise(p);
        sum += factor;
        factor /= 2.0f;
        if (i < level - 1) {
            p = m_m * p * (2.0f + (float) i * 0.01f);
        }
    }
    return f / sum;
}

glm::vec3 PlantArchitect::FBMField::Get3(const glm::vec3 &in, unsigned int level) {
    return {Get(in, level), Get(in + glm::vec3(0.25, 0.57, 0.14), level),
            Get(in + glm::vec3(0.63, 0.23, 0.56), level)};
}

glm::vec3 PlantArchitect::FBMField::GetT(const glm::vec3 &q, float t, float frequency, float density, int level) {
    auto q1 = frequency * glm::sin(t + glm::simplex(q));
    return Get3(density * (q + q1), level);
}
