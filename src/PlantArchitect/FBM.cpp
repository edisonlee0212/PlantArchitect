//
// Created by lllll on 12/28/2021.
//

#include "FBM.hpp"

void PlantArchitect::FBM::OnCreate() {
}


void PlantArchitect::FBM::OnInspect() {
    static bool draw = true;
    ImGui::Checkbox("Render field", &draw);
    if (draw) {
        static auto minRange = glm::vec3(-25, 0, -25);
        static auto maxRange = glm::vec3(25, 30, 25);
        static float step = 3.0f;
        if (ImGui::DragFloat3("Min", &minRange.x, 0.1f)) {
            minRange = (glm::min) (minRange, maxRange);
        }
        if (ImGui::DragFloat3("Max", &maxRange.x, 0.1f)) {
            maxRange = (glm::max) (minRange, maxRange);
        }
        ImGui::DragFloat("Step", &step, 0.01f);
        step = glm::clamp(step, 0.1f, 10.0f);

        const int sx = (int) ((maxRange.x - minRange.x + step) / step);
        const int sy = (int) ((maxRange.y - minRange.y + step) / step);
        const int sz = (int) ((maxRange.z - minRange.z + step) / step);
        auto voxelSize = sx * sy * sz;


        static glm::vec3 frequency = {10.0f, 0.0f, 0.0f};
        static glm::vec3 density = {0.02f, 1.0f, 1.f};
        static glm::vec3 factor = {1.0f, 0.0f, 0.0f};
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

        static bool useColor = false;
        ImGui::Checkbox("Color Cube", &useColor);

        if(!useColor) {
            static float lineWidth = 0.05f;
            static float pointSize = 0.1f;
            static std::vector<glm::vec3> starts;
            static std::vector<glm::vec3> ends;
            static std::vector<glm::mat4> matrices;
            static glm::vec4 color = {0.0f, 1.0f, 0.0f, 0.5f};
            static glm::vec4 pointColor = {1.0f, 0.0f, 0.0f, 0.75f};
            starts.resize(voxelSize);
            ends.resize(voxelSize);
            matrices.resize(voxelSize);
            ImGui::DragFloat("Width", &lineWidth, 0.01f);
            ImGui::DragFloat("Point Size", &pointSize, 0.01f);
            ImGui::ColorEdit4("Vector Color", &color.x);
            ImGui::ColorEdit4("Point Color", &pointColor.x);
            std::vector<std::shared_future<void>> results;
            Jobs::ParallelFor(voxelSize, [&](unsigned i) {
                float z = (i % sz) * step + minRange.z;
                float y = ((i / sz) % sy) * step + minRange.y;
                float x = ((i / sz / sy) % sx) * step + minRange.x;
                glm::vec3 start = {x, y, z};
                starts[i] = start;
                /*
                colors[i] = {factor.x * GetT(coord, t, frequency.x, density.x, 6) +
                             factor.y * GetT(coord, t, frequency.y, density.y, 6) +
                             factor.z * GetT(coord, t, frequency.z, density.z, 6), alpha};
                             */
                glm::vec3 v = {GetT({x, y, z}, t, frequency.x, density.x, 6)};
                ends[i] = start + v * 2.0f - glm::vec3(1);
                matrices[i] = glm::translate(glm::vec3(x, y, z)) * glm::scale(glm::vec3(pointSize));
            }, results);
            for (const auto &i: results) i.wait();
            Graphics::DrawGizmoRays(color, starts, ends, lineWidth);
            Graphics::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube, pointColor, matrices);
        } else {
            static float width = 0.3f;
            static float alpha = 0.2f;
            static std::vector<glm::vec4> colors;
            static std::vector<glm::mat4> matrices;
            colors.resize(voxelSize);
            matrices.resize(voxelSize);

            ImGui::DragFloat("Cube size", &width, 0.01f);
            ImGui::DragFloat("Alpha", &alpha, 0.01f, 0.0f, 1.0f);
            std::vector<std::shared_future<void>> results;
            Jobs::ParallelFor(voxelSize, [&](unsigned i) {
                float z = (i % sz) * step + minRange.z;
                float y = ((i / sz) % sy) * step + minRange.y;
                float x = ((i / sz / sy) % sx) * step + minRange.x;
                glm::vec3 coord = {x, y, z};
                /*
                colors[i] = {factor.x * GetT(coord, t, frequency.x, density.x, 6) +
                             factor.y * GetT(coord, t, frequency.y, density.y, 6) +
                             factor.z * GetT(coord, t, frequency.z, density.z, 6), alpha};
                             */
                colors[i] = {GetT(coord, t, frequency.x, density.x, 6), alpha};
                matrices[i] = glm::translate(glm::vec3(x, y, z)) * glm::scale(glm::vec3(width));
            }, results);
            for (const auto &i: results) i.wait();
            /*
            int i = 0;
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
            */
            Graphics::DrawGizmoMeshInstancedColored(DefaultResources::Primitives::Cube, colors, matrices);
        }
    }
}

float PlantArchitect::FBM::Get(const glm::vec3 &in, unsigned int octaves) {
    float f = 0;
    float sum = 0.0f;
    float factor = 0.5f;
    glm::vec3 p = in;
    for (unsigned i = 0; i < octaves; i++) {
        f += factor * glm::simplex(p);
        sum += factor;
        factor /= 2.0f;
        if (i < octaves - 1) {
            p = m_m * p * (2.0f + (float) i * 0.01f);
        }
    }
    return f / sum;
}

glm::vec3 PlantArchitect::FBM::Get3(const glm::vec3 &in, unsigned int level) {
    return {Get(in, level), Get(in + glm::vec3(0.25, 0.57, 0.14), level),
            Get(in + glm::vec3(0.63, 0.23, 0.56), level)};
}

glm::vec3 PlantArchitect::FBM::GetT(const glm::vec3 &q, float t, float frequency, float density, int level) {
    auto q1 = frequency * glm::sin(t + glm::simplex(q));
    return Get3(density * (q + q1), level);
}
