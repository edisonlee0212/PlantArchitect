//
// Created by lllll on 9/5/2021.
//

#include "DepthCamera.hpp"

using namespace PlantArchitect;

std::shared_ptr<OpenGLUtils::GLProgram> DepthCamera::m_depthTransferProgram;
std::shared_ptr<OpenGLUtils::GLVAO> DepthCamera::m_depthTransferVAO;

void DepthCamera::OnInspect() {
    ImGui::Checkbox("Use Camera Resolution", &m_useCameraResolution);
    if(!m_useCameraResolution){
        ImGui::DragInt2("Resolution", &m_resX);
    }

    if (ImGui::TreeNode("Content"))
    {
        static float debugSacle = 0.25f;
        ImGui::DragFloat("Scale", &debugSacle, 0.01f, 0.1f, 1.0f);
        debugSacle = glm::clamp(debugSacle, 0.1f, 1.0f);
        ImGui::Image(
                (ImTextureID)m_colorTexture->UnsafeGetGLTexture()->Id(),
                ImVec2(m_resolutionX * debugSacle, m_resolutionY * debugSacle),
                ImVec2(0, 1),
                ImVec2(1, 0));
        ImGui::TreePop();
    }

    FileUtils::SaveFile("Screenshot", "Texture2D", {".png", ".jpg"}, [this](const std::filesystem::path &filePath) {
        m_colorTexture->SetPathAndSave(ProjectManager::GetRelativePath(filePath));
    });
}
void DepthCamera::Update() {
    if (!GetOwner().HasPrivateComponent<Camera>())
        return;
    auto cameraComponent = GetOwner().GetOrSetPrivateComponent<Camera>().lock();
    // 1. Resize to resolution
    auto resolution = cameraComponent->GetResolution();
    if(m_useCameraResolution) {
        m_resX = resolution.x;
        m_resY = resolution.y;
    }else{
        int x = resolution.x;
        int y = resolution.y;
        m_resX = glm::clamp(m_resX, 1, x);
        m_resY = glm::clamp(m_resY, 1, y);
    }
    if (m_resolutionX != m_resX || m_resY) {
        m_resolutionX = m_resX;
        m_resolutionY = m_resY;
        m_colorTexture->UnsafeGetGLTexture()->ReSize(
                0, GL_RGB32F, GL_RGB, GL_FLOAT, 0, m_resolutionX, m_resolutionY);
    }
    // 2. Render to depth texture
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    OpenGLUtils::SetEnable(OpenGLCapability::CullFace, false);
    m_depthTransferVAO->Bind();

    m_depthTransferProgram->Bind();

    AttachTexture(m_colorTexture->UnsafeGetGLTexture().get(), GL_COLOR_ATTACHMENT0);
    Bind();
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    cameraComponent->GetDepthStencil()->UnsafeGetGLTexture()->Bind(0);
    m_depthTransferProgram->SetInt("depthStencil", 0);
    m_depthTransferProgram->SetFloat("near", cameraComponent->m_nearDistance);
    m_depthTransferProgram->SetFloat("far", cameraComponent->m_farDistance);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}
void DepthCamera::OnCreate() {
    if (!m_depthTransferProgram) {
        auto fragShaderCode =
                std::string("#version 450 core\n") +
                FileUtils::LoadFileAsString(std::filesystem::path("./PlantArchitectResources") /
                                            "Shaders/Fragment/DepthCopy.frag");
        auto fragShader = AssetManager::CreateAsset<OpenGLUtils::GLShader>();
        fragShader->Set(OpenGLUtils::ShaderType::Fragment, fragShaderCode);
        m_depthTransferProgram =
                AssetManager::CreateAsset<OpenGLUtils::GLProgram>();
        m_depthTransferProgram->Link(
                DefaultResources::GLShaders::TexturePassThrough, fragShader);
    }

    if(!m_depthTransferVAO){
        float quadVertices[] = {
                // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
                // positions   // texCoords
                -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f,

                -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 0.0f, 1.0f, 1.0f,  1.0f, 1.0f};

        m_depthTransferVAO = std::make_shared<OpenGLUtils::GLVAO>();
        m_depthTransferVAO->SetData(sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        m_depthTransferVAO->EnableAttributeArray(0);
        m_depthTransferVAO->SetAttributePointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
        m_depthTransferVAO->EnableAttributeArray(1);
        m_depthTransferVAO->SetAttributePointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    }

    m_resolutionX = 1;
    m_resolutionY = 1;

    m_colorTexture = AssetManager::CreateAsset<Texture2D>();
    m_colorTexture->m_name = "CameraTexture";
    m_colorTexture->UnsafeGetGLTexture() = std::make_shared<OpenGLUtils::GLTexture2D>(
            0, GL_RGB32F, m_resolutionX, m_resolutionY, false);
    m_colorTexture->UnsafeGetGLTexture()->SetData(0, GL_RGB32F, GL_RGB, GL_FLOAT, 0);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    m_colorTexture->UnsafeGetGLTexture()->SetInt(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    AttachTexture(m_colorTexture->UnsafeGetGLTexture().get(), GL_COLOR_ATTACHMENT0);
}

DepthCamera &DepthCamera::operator=(const DepthCamera & source) {
    m_useCameraResolution = source.m_useCameraResolution;
    m_resX = source.m_resX;
    m_resY = source.m_resY;
    m_colorTexture = source.m_colorTexture;
    return *this;
}
