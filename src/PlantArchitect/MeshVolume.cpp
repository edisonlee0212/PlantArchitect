#include <MeshVolume.hpp>
#include "Graphics.hpp"
#include "DefaultResources.hpp"
#ifdef RAYTRACERFACILITY
#include "RayTracerLayer.hpp"
#include <MLVQRenderer.hpp>
using namespace RayTracerFacility;
#endif
using namespace PlantArchitect;


void MeshVolume::OnCreate() {

    SetEnabled(true);
}

void MeshVolume::OnInspect() {
    IVolume::OnInspect();
    Editor::DragAndDropButton<MeshRenderer>(m_meshRendererRef, "Mesh Renderer");
    auto meshRenderer = m_meshRendererRef.Get<MeshRenderer>();
    if (meshRenderer) {
        if (m_displayBounds) {
            const auto globalTransform = meshRenderer->GetScene()->GetDataComponent<GlobalTransform>(
                    meshRenderer->GetOwner());
            Gizmos::DrawGizmoMesh(
                    meshRenderer->m_mesh.Get<Mesh>(), glm::vec4(0, 1, 0, 0.2),
                    globalTransform.m_value,
                    1);
        }
    }
}

bool MeshVolume::InVolume(const glm::vec3 &position) {
    auto meshRenderer = m_meshRendererRef.Get<MeshRenderer>();
    if (!meshRenderer) return false;
    auto mesh = meshRenderer->m_mesh.Get<Mesh>();
    if(!mesh) return false;
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
#ifdef RAYTRACERFACILITY
    std::vector<PointCloudSample> pcSamples;
    PointCloudSample pcSample;
    pcSample.m_start = position;
    pcSample.m_direction = glm::vec3(1, 0, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(-1, 0, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, 1, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, -1, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, 0, 1);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, 0, -1);
    pcSamples.emplace_back(pcSample);

    CudaModule::SamplePointCloud(
            Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
            pcSamples);
    for(const auto& i : pcSamples){
        if(!i.m_hit || i.m_handle != meshRenderer->GetHandle()) return false;
    }
    return true;

#else
    return false;
#endif
}


bool MeshVolume::InVolume(const GlobalTransform &globalTransform,
                          const glm::vec3 &position) {
    auto meshRenderer = m_meshRendererRef.Get<MeshRenderer>();
    if (!meshRenderer) return false;
    auto mesh = meshRenderer->m_mesh.Get<Mesh>();
    if(!mesh) return false;
#ifdef RAYTRACERFACILITY
    std::vector<PointCloudSample> pcSamples;
    PointCloudSample pcSample;
    pcSample.m_start = position;
    pcSample.m_direction = glm::vec3(1, 0, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(-1, 0, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, 1, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, -1, 0);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, 0, 1);
    pcSamples.emplace_back(pcSample);
    pcSample.m_direction = glm::vec3(0, 0, -1);
    pcSamples.emplace_back(pcSample);

    CudaModule::SamplePointCloud(
            Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
            pcSamples);
    for(const auto& i : pcSamples){
        if(!i.m_hit || i.m_handle != meshRenderer->GetHandle()) return false;
    }
    return true;
#else
    return false;
#endif
}

void MeshVolume::InVolume(const GlobalTransform &globalTransform, const std::vector<glm::vec3> &positions,
                          std::vector<bool> &results) {
    if(positions.empty()) return;
    auto meshRenderer = m_meshRendererRef.Get<MeshRenderer>();
    if (!meshRenderer) return;
    auto mesh = meshRenderer->m_mesh.Get<Mesh>();
    if(!mesh) return;
#ifdef RAYTRACERFACILITY
    std::vector<PointCloudSample> pcSamples;
    for(int i = 0; i < positions.size(); i++) {
        PointCloudSample pcSample;
        pcSample.m_start = positions[i];
        pcSample.m_direction = glm::vec3(1, 0, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(-1, 0, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, 1, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, -1, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, 0, 1);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, 0, -1);
        pcSamples.emplace_back(pcSample);
    }
    CudaModule::SamplePointCloud(
            Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
            pcSamples);
    for(int i = 0; i < positions.size(); i++) {
        results[i] = true;
        for(int j = 0; j < 6; j++){
            int index = i * 6 + j;
            if(!pcSamples[index].m_hit || pcSamples[index].m_handle != meshRenderer->GetHandle()) {
                results[i] = false;
                break;
            }
        }
    }
#endif
}

void MeshVolume::InVolume(const std::vector<glm::vec3> &positions, std::vector<bool> &results) {
    if(positions.empty()) return;
    auto meshRenderer = m_meshRendererRef.Get<MeshRenderer>();
    if (!meshRenderer) return;
    auto mesh = meshRenderer->m_mesh.Get<Mesh>();
    if(!mesh) return;
#ifdef RAYTRACERFACILITY
    std::vector<PointCloudSample> pcSamples;
    for(int i = 0; i < positions.size(); i++) {
        PointCloudSample pcSample;
        pcSample.m_start = positions[i];
        pcSample.m_direction = glm::vec3(1, 0, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(-1, 0, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, 1, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, -1, 0);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, 0, 1);
        pcSamples.emplace_back(pcSample);
        pcSample.m_direction = glm::vec3(0, 0, -1);
        pcSamples.emplace_back(pcSample);
    }
    CudaModule::SamplePointCloud(
            Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
            pcSamples);
    for(int i = 0; i < positions.size(); i++) {
        results[i] = true;
        for(int j = 0; j < 6; j++){
            int index = i * 6 + j;
            if(!pcSamples[index].m_hit || pcSamples[index].m_handle != meshRenderer->GetHandle()) {
                results[i] = false;
                break;
            }
        }
    }
#endif
}

glm::vec3 MeshVolume::GetRandomPoint() {
    auto meshRenderer = m_meshRendererRef.Get<MeshRenderer>();
    if (!meshRenderer) return {0, 0, 0};
    auto mesh = meshRenderer->m_mesh.Get<Mesh>();
    if (!mesh) return {0, 0, 0};
    const auto globalTransform = GetScene()->GetDataComponent<GlobalTransform>(GetOwner());
    auto randomPoint = glm::linearRand(globalTransform.m_value * glm::vec4(mesh->GetBound().m_min, 1.0f),
                                       globalTransform.m_value * glm::vec4(mesh->GetBound().m_max, 1.0f));
    bool inVolume = InVolume(randomPoint);
    while(!inVolume){
        randomPoint = glm::linearRand(globalTransform.m_value * glm::vec4(mesh->GetBound().m_min, 1.0f),
                                      globalTransform.m_value * glm::vec4(mesh->GetBound().m_max, 1.0f));
        inVolume = InVolume(randomPoint);
    }
    return randomPoint;
}

void MeshVolume::Serialize(YAML::Emitter &out) {
    IVolume::Serialize(out);
    m_meshRendererRef.Save("m_meshRendererRef", out);
}

void MeshVolume::Deserialize(const YAML::Node &in) {
    IVolume::Deserialize(in);
    m_meshRendererRef.Load("m_meshRendererRef", in, GetScene());
}

void MeshVolume::OnDestroy() {
    IVolume::OnDestroy();
    m_meshRendererRef.Clear();
}

void MeshVolume::Relink(const std::unordered_map<Handle, Handle> &map, const std::shared_ptr<Scene> &scene) {
    m_meshRendererRef.Relink(map, scene);
}


