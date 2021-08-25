#include <RayTracedRenderer.hpp>

using namespace RayTracerFacility;

#include <EditorManager.hpp>
#include <MeshRenderer.hpp>

using namespace UniEngine;

void RayTracedRenderer::OnGui() {
    ImGui::DragFloat("Metallic##RayTracedRenderer", &m_metallic, 0.01f, 0.0f,
                     1.0f);
    ImGui::DragFloat("Roughness##RayTracedRenderer", &m_roughness, 0.01f, 0.0f,
                     1.0f);
    ImGui::DragFloat("Transparency##RayTracedRenderer", &m_transparency, 0.01f,
                     0.0f, 1.0f);
    ImGui::DragFloat("Diffuse intensity##RayTracedRenderer", &m_diffuseIntensity,
                     0.01f, 0.0f, 100.0f);
    ImGui::ColorEdit3("Surface Color##RayTracedRenderer", &m_surfaceColor.x);
    EditorManager::DragAndDropButton<Mesh>(m_mesh, "Mesh");
    ImGui::Text("Material: ");
    ImGui::SameLine();

    if (ImGui::TreeNode("Textures##RayTracerMaterial")) {
        EditorManager::DragAndDropButton<Texture2D>(m_albedoTexture, "Albedo");
        EditorManager::DragAndDropButton<Texture2D>(m_normalTexture, "Normal");

        ImGui::TreePop();
    }

}

void RayTracedRenderer::SyncWithMeshRenderer() {
    Entity owner = GetOwner();
    if (owner.HasPrivateComponent<MeshRenderer>()) {
        auto mmr = owner.GetOrSetPrivateComponent<MeshRenderer>().lock();
        m_mesh = mmr->m_mesh;
        auto mat = mmr->m_material.Get<Material>();
        m_roughness = mat->m_roughness;
        m_metallic = mat->m_metallic;
        m_surfaceColor = mat->m_albedoColor;
    }
}

void RayTracedRenderer::Clone(
        const std::shared_ptr<IPrivateComponent> &target) {
    *this = *std::static_pointer_cast<RayTracedRenderer>(target);
}

void RayTracedRenderer::Serialize(YAML::Emitter &out) {
    out << YAML::Key << "m_diffuseIntensity" << YAML::Value << m_diffuseIntensity;
    out << YAML::Key << "m_transparency" << YAML::Value << m_transparency;
    out << YAML::Key << "m_metallic" << YAML::Value << m_metallic;
    out << YAML::Key << "m_roughness" << YAML::Value << m_roughness;
    out << YAML::Key << "m_surfaceColor" << YAML::Value << m_surfaceColor;

    m_mesh.Save("m_mesh", out);
    m_albedoTexture.Save("m_albedoTexture", out);
    m_normalTexture.Save("m_normalTexture", out);
}

void RayTracedRenderer::Deserialize(const YAML::Node &in) {
    m_diffuseIntensity = in["m_diffuseIntensity"].as<float>();
    m_transparency = in["m_transparency"].as<float>();
    m_metallic = in["m_metallic"].as<float>();
    m_roughness = in["m_roughness"].as<float>();
    m_surfaceColor = in["m_surfaceColor"].as<glm::vec3>();

    m_mesh.Load("m_mesh", in);
    m_albedoTexture.Load("m_albedoTexture", in);
    m_normalTexture.Load("m_normalTexture", in);
}

void RayTracedRenderer::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_mesh);
    list.push_back(m_albedoTexture);
    list.push_back(m_normalTexture);
}