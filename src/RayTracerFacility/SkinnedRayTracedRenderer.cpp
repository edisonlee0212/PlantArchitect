//
// Created by lllll on 9/2/2021.
//

#include "SkinnedRayTracedRenderer.hpp"

using namespace RayTracerFacility;

#include <EditorManager.hpp>
using namespace UniEngine;

void SkinnedRayTracedRenderer::OnGui() {
    if(ImGui::Button("Sync")) Sync();

    ImGui::DragFloat("Metallic##RayTracedRenderer", &m_metallic, 0.01f, 0.0f,
                     1.0f);
    ImGui::DragFloat("Roughness##RayTracedRenderer", &m_roughness, 0.01f, 0.0f,
                     1.0f);
    ImGui::DragFloat("Transparency##RayTracedRenderer", &m_transparency, 0.01f,
                     0.0f, 1.0f);
    ImGui::DragFloat("Diffuse intensity##RayTracedRenderer", &m_diffuseIntensity,
                     0.01f, 0.0f, 100.0f);
    ImGui::ColorEdit3("Surface Color##RayTracedRenderer", &m_surfaceColor.x);
    EditorManager::DragAndDropButton<SkinnedMeshRenderer>(m_skinnedMeshRenderer, "SkinnedMeshRenderer");
    ImGui::Text("Material: ");
    ImGui::SameLine();

    if (ImGui::TreeNode("Textures##RayTracerMaterial")) {
        EditorManager::DragAndDropButton<Texture2D>(m_albedoTexture, "Albedo");
        EditorManager::DragAndDropButton<Texture2D>(m_normalTexture, "Normal");

        ImGui::TreePop();
    }

}

void SkinnedRayTracedRenderer::Sync() {
    Entity owner = GetOwner();
    if (owner.HasPrivateComponent<SkinnedMeshRenderer>()) {
        auto mmr = owner.GetOrSetPrivateComponent<SkinnedMeshRenderer>().lock();
        m_skinnedMeshRenderer = mmr;
        auto mat = mmr->m_material.Get<Material>();
        m_roughness = mat->m_roughness;
        m_metallic = mat->m_metallic;
        m_surfaceColor = mat->m_albedoColor;
    }
}

void SkinnedRayTracedRenderer::Clone(
        const std::shared_ptr<IPrivateComponent> &target) {
    *this = *std::static_pointer_cast<SkinnedRayTracedRenderer>(target);
}

void SkinnedRayTracedRenderer::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_diffuseIntensity" << YAML::Value << m_diffuseIntensity;
    out << YAML::Key << "m_transparency" << YAML::Value << m_transparency;
    out << YAML::Key << "m_metallic" << YAML::Value << m_metallic;
    out << YAML::Key << "m_roughness" << YAML::Value << m_roughness;
    out << YAML::Key << "m_surfaceColor" << YAML::Value << m_surfaceColor;

    m_skinnedMeshRenderer.Save("m_skinnedMeshRenderer", out);
    m_albedoTexture.Save("m_albedoTexture", out);
    m_normalTexture.Save("m_normalTexture", out);
}

void SkinnedRayTracedRenderer::Deserialize(const YAML::Node &in) {
    m_diffuseIntensity = in["m_diffuseIntensity"].as<float>();
    m_transparency = in["m_transparency"].as<float>();
    m_metallic = in["m_metallic"].as<float>();
    m_roughness = in["m_roughness"].as<float>();
    m_surfaceColor = in["m_surfaceColor"].as<glm::vec3>();

    m_skinnedMeshRenderer.Load("m_skinnedMeshRenderer", in);
    m_albedoTexture.Load("m_albedoTexture", in);
    m_normalTexture.Load("m_normalTexture", in);
}

void SkinnedRayTracedRenderer::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_albedoTexture);
    list.push_back(m_normalTexture);
}

void SkinnedRayTracedRenderer::Relink(const std::unordered_map<Handle, Handle> &map){
    m_skinnedMeshRenderer.Relink(map);
}