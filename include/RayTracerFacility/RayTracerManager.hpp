#pragma once
#include <UniEngine-pch.hpp>
#include <Application.hpp>
#include <CUDAModule.hpp>
#include <Cubemap.hpp>
#include <EditorManager.hpp>
#include <InputManager.hpp>
#include <MeshRenderer.hpp>
#include <AssetManager.hpp>

#include <WindowManager.hpp>
#include <memory>
#include <ray_tracer_facility_export.h>
using namespace UniEngine;
namespace RayTracerFacility {
class RayTracerRenderWindow {
public:
  std::string m_name;
  bool m_renderingEnabled = true;
  float m_lastX = 0;
  float m_lastY = 0;
  float m_lastScrollY = 0;
  bool m_startMouse = false;
  bool m_startScroll = false;
  bool m_rightMouseButtonHold = false;

  float m_resolutionMultiplier = 0.5f;
  std::unique_ptr<OpenGLUtils::GLTexture2D> m_output;
  glm::ivec2 m_outputSize = glm::ivec2(1024, 1024);
  bool m_rendered = false;
  void Init(const std::string &name);
  [[nodiscard]] glm::ivec2 Resize() const;
  void OnGui();
};

class RAY_TRACER_FACILITY_API RayTracerManager {
protected:
#pragma region Class related
  RayTracerManager() = default;
  RayTracerManager(RayTracerManager &&) = default;
  RayTracerManager(const RayTracerManager &) = default;
  RayTracerManager &operator=(RayTracerManager &&) = default;
  RayTracerManager &operator=(const RayTracerManager &) = default;
#pragma endregion
public:
  std::shared_ptr<Cubemap> m_environmentalMap;

  DefaultRenderingProperties m_defaultRenderingProperties;
  RayTracerRenderWindow m_defaultWindow;

  void UpdateScene() const;

  static RayTracerManager &GetInstance();
  static void Init();
  static void Update();
  static void OnGui();
  static void End();
};
} // namespace RayTracerFacility