#include <CUDAModule.hpp>
#include <Optix7.hpp>
#include <RayTracerUtilities.cuh>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <RayDataDefinations.hpp>

#include "DisneyBssrdf.hpp"

namespace RayTracerFacility {
extern "C" __constant__ DefaultRenderingLaunchParams
    defaultRenderingLaunchParams;
struct DefaultRenderingRadianceRayData {
  unsigned m_hitCount;
  Random m_random;
  glm::vec3 m_energy;
  glm::vec3 m_pixelNormal;
  glm::vec3 m_pixelAlbedo;
};
struct DefaultRenderingSpSamplerRayData {
  unsigned m_instanceId;
  glm::vec3 m_p0;
  glm::vec3 m_p1;
  glm::vec3 m_n1;
  bool m_found = false;
};
#pragma region Closest hit functions
#pragma region Helpers

// This function is purely for the function below, it clearly doesn't handle all
// corner cases of input.
static __forceinline__ __device__ double pow_frac(const double x, const int n) {
  return (n == 0) ? 1.0 : pow_frac(x, n - 1) * x / (double)n;
}

// This is a taylor approximation of exp function, might not be super accurate.
static __forceinline__ __device__ double
exp_compile_fractional(const double x) {
  return pow_frac(x, 0) + pow_frac(x, 1) + pow_frac(x, 2) + pow_frac(x, 3) +
         pow_frac(x, 4) + pow_frac(x, 5) + pow_frac(x, 6) + pow_frac(x, 7) +
         pow_frac(x, 8) + pow_frac(x, 9);
}

// Log(n) time solution to acquire pow(e,n), where n is an integer.
static __forceinline__ __device__ double exp_compile_integer(const int n) {
  // e (mathematical constant)
  // https://en.wikipedia.org/wiki/E_(mathematical_constant)
  constexpr double e = 2.71828;
  if (n == 0)
    return 1.0;

  const auto half_x = n / 2;
  const auto ex = exp_compile_integer(half_x);
  return ex * ex * ((n % 2) ? e : 1.0);
}

static __forceinline__ __device__ double exp_compile(const double x) {
  if (x < 0.0)
    return 1.0 / exp_compile(-x);

  const int ix = (int)(x);  // floor is not supported in compile time.
  const double fx = x - ix; // It should always be between 0 and 1.
  return exp_compile_fractional(fx) * exp_compile_integer(ix);
}

static __forceinline__ __device__ glm::vec3
Sr(const float &radius, const glm::vec3 &d, const glm::vec3 &R) {
  const auto r = (radius < 0.000001f) ? 0.000001f : radius;
  const auto EIGHT_PI = 8.0f * glm::pi<float>();
  return R *
         (glm::exp(glm::vec3(-r) / d) + glm::exp(glm::vec3(-r) / (3.0f * d))) /
         (EIGHT_PI * d * r);
}

static __forceinline__ __device__ float
Pdf_Sr(const int &ch, const glm::vec3 &d, const float &radius) {
  const auto EIGHT_PI = 8.0f * glm::pi<float>();
  const auto r = (radius < 0.000001f) ? 0.000001f : radius;
  return (exp(-r / d[ch]) + exp(-r / (3.0f * d[ch]))) / (EIGHT_PI * d[ch] * r) *
         (1.0f / (0.25f * (4.0f - exp_compile(-16.0f) -
                           3.0f * exp_compile(-16.0f / 3.0f))));
}

#pragma endregion

static __forceinline__ __device__ void
Sample_S(const DisneyBssrdf &disneyBssrdf, const glm::vec3 &p0,
         const glm::vec3 &in, const glm::vec3 &nn, const glm::vec3 &t0,
         Random &random, glm::vec3 &p1, glm::vec3 &n1, glm::vec3 &weight) {
  auto &d = disneyBssrdf.d;
  auto &R = disneyBssrdf.R;
  const auto r0 = random();
  glm::vec3 vx, vy, vz;
  const auto btn = glm::normalize(glm::cross(nn, t0));
  const auto tn = glm::normalize(glm::cross(btn, nn));
  if (r0 < 0.5f) {
    vx = btn;
    vy = nn;
    vz = tn;
  } else if (r0 < 0.75f) {
    vx = tn;
    vy = btn;
    vz = nn;
  } else {
    vx = nn;
    vy = tn;
    vz = btn;
  }

#pragma region Sample_Ch
  const int channels = 3;
  const auto ch = glm::clamp((int)(random() * channels), 0, channels - 1);
#pragma endregion
  const auto tmp = random();
#pragma region Sample_Sr
  constexpr auto quater_cutoff = 0.25f;

  // importance sampling burley profile
  const auto ret = (tmp < quater_cutoff)
                       ? -d[ch] * log(4.0f * tmp)
                       : -3.0f * d[ch] * log((tmp - quater_cutoff) * 1.3333f);

  // ignore all samples outside the sampling range
  const auto r = (ret > 16.0f * d[ch]) ? -1.0f : ret;
#pragma endregion
  assert(r > -0.01f);
  const auto rMax = fmax(0.0015f, 16.0f * d[ch]);
  const auto l = 2.0f * sqrt(rMax * rMax - r * r);

  const auto phi = glm::pi<float>() * 2.0f * random();
  const auto source = p0 + r * (vx * cos(phi) + vz * sin(phi)) + l * vy * 0.5f;
  DefaultRenderingSpSamplerRayData perRayData;
  perRayData.m_p0 = p0;
  perRayData.m_instanceId = optixGetInstanceId();
  perRayData.m_found = false;
  uint32_t u0, u1;
  PackRayDataPointer(&perRayData, u0, u1);

  optixTrace(
      defaultRenderingLaunchParams.m_traversable,
      make_float3(source.x, source.y, source.z), make_float3(vy.x, vy.y, vy.z),
      1e-3f, // tmin
      l,     // tmax
      0.0f,  // rayTime
      static_cast<OptixVisibilityMask>(255), OPTIX_RAY_FLAG_NONE,
      static_cast<int>(DefaultRenderingRayType::SampleSpRayType), // SBT offset
      static_cast<int>(DefaultRenderingRayType::RayTypeCount),    // SBT stride
      static_cast<int>(
          DefaultRenderingRayType::SampleSpRayType), // missSBTIndex
      u0, u1);
  if (perRayData.m_found) {
#pragma region BSSRDF
    const auto bssrdf = Sr(glm::distance(perRayData.m_p1, p0), d, R);
#pragma endregion
#pragma region PDF
    glm::vec3 diff = p0 - perRayData.m_p1;
    glm::vec3 n = perRayData.m_n1;
    glm::vec3 dLocal(dot(btn, diff), dot(nn, diff), dot(tn, diff));
    glm::vec3 nLocal(dot(btn, n), dot(nn, n), dot(tn, n));

    float rProj[3] = {sqrt(dLocal.y * dLocal.y + dLocal.z * dLocal.z),
                      sqrt(dLocal.x * dLocal.x + dLocal.z * dLocal.z),
                      sqrt(dLocal.x * dLocal.x + dLocal.y * dLocal.y)};

    constexpr float axisProb[3] = {0.25f, 0.5f, 0.25f};
    auto pdf = 0.0f;
    for (auto axis = 0; axis < 3; ++axis) {
      for (auto ch = 0; ch < 3; ++ch) {
        pdf += Pdf_Sr(ch, d, rProj[axis]) * std::abs(nLocal[axis]) *
               axisProb[axis];
      }
    }
    pdf /= channels;
#pragma endregion
    weight = bssrdf / pdf;
  } else {
    weight = glm::vec3(0.0f);
  }
}

extern "C" __global__ void __closesthit__radiance() {
  const float3 rayDirectionInternal = optixGetWorldRayDirection();
  glm::vec3 rayDirection = glm::vec3(
      rayDirectionInternal.x, rayDirectionInternal.y, rayDirectionInternal.z);
#pragma region Retrive information
  const auto &sbtData = *(const DefaultSbtData *)optixGetSbtDataPointer();
  const float2 triangleBarycentricsInternal = optixGetTriangleBarycentrics();
  const int primitiveId = optixGetPrimitiveIndex();

  auto indices = sbtData.m_mesh.GetIndices(primitiveId);
  auto texCoord =
      sbtData.m_mesh.GetTexCoord(triangleBarycentricsInternal, indices);
  auto normal =
      sbtData.m_mesh.GetNormal(triangleBarycentricsInternal, indices);
  if (glm::dot(rayDirection, normal) > 0.0f) {
    normal = -normal;
  }
  auto tangent =
      sbtData.m_mesh.GetTangent(triangleBarycentricsInternal, indices);
  auto hitPoint =
      sbtData.m_mesh.GetPosition(triangleBarycentricsInternal, indices);
#pragma endregion
  DefaultRenderingRadianceRayData &perRayData =
      *GetRayDataPointer<DefaultRenderingRadianceRayData>();
  unsigned hitCount = perRayData.m_hitCount + 1;

  if (defaultRenderingLaunchParams.m_defaultRenderingProperties
          .m_enableGround &&
      hitPoint.y < defaultRenderingLaunchParams
                       .m_defaultRenderingProperties.m_groundHeight) {
    // start with some ambient term
    auto energy = glm::vec3(0.0f);
    uint32_t u0, u1;
    PackRayDataPointer(&perRayData, u0, u1);
    perRayData.m_hitCount = hitCount;
    perRayData.m_energy = glm::vec3(0.0f);
    normal = glm::vec3(0, 1, 0);
    float metallic = defaultRenderingLaunchParams.m_defaultRenderingProperties
                         .m_groundMetallic;
    metallic =
        (metallic == 1.0f ? -1.0f : 1.0f / glm::pow(1.0f - metallic, 3.0f));
    float roughness = defaultRenderingLaunchParams.m_defaultRenderingProperties
                          .m_groundRoughness;
    glm::vec3 albedoColor =
        defaultRenderingLaunchParams.m_defaultRenderingProperties.m_groundColor;
    if (perRayData.m_hitCount <=
        defaultRenderingLaunchParams.m_defaultRenderingProperties
            .m_bounceLimit) {
      energy = glm::vec3(0.0f);
      float f = 1.0f;
      if (metallic >= 0.0f)
        f = (metallic + 2) / (metallic + 1);
      float3 rayOrigin = optixGetWorldRayOrigin();
      glm::vec3 rayOrig = glm::vec3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
      hitPoint =
          rayOrig -
          rayDirection *
              ((rayOrig.y - defaultRenderingLaunchParams
                                .m_defaultRenderingProperties.m_groundHeight) /
               rayDirection.y);
      float3 incidentRayOrigin;
      float3 newRayDirectionInternal;
      BRDF(metallic, perRayData.m_random, normal, hitPoint, rayDirection,
           incidentRayOrigin, newRayDirectionInternal);
      optixTrace(
          defaultRenderingLaunchParams.m_traversable, incidentRayOrigin,
          newRayDirectionInternal,
          1e-3f, // tmin
          1e20f, // tmax
          0.0f,  // rayTime
          static_cast<OptixVisibilityMask>(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          static_cast<int>(
              DefaultRenderingRayType::RadianceRayType),           // SBT offset
          static_cast<int>(DefaultRenderingRayType::RayTypeCount), // SBT stride
          static_cast<int>(
              DefaultRenderingRayType::RadianceRayType), // missSBTIndex
          u0, u1);
      energy +=
          albedoColor *
          glm::clamp(
              glm::abs(glm::dot(normal, glm::vec3(newRayDirectionInternal.x,
                                                  newRayDirectionInternal.y,
                                                  newRayDirectionInternal.z))) *
                      roughness +
                  (1.0f - roughness) * f,
              0.0f, 1.0f) *
          perRayData.m_energy;
    }
    if (hitCount == 1) {
      perRayData.m_pixelNormal = normal;
      perRayData.m_pixelAlbedo = albedoColor;
    }
    perRayData.m_energy = energy + 0.0f * albedoColor;
  } else {


    // start with some ambient term
    auto energy = glm::vec3(0.0f);
    uint32_t u0, u1;
    PackRayDataPointer(&perRayData, u0, u1);
    perRayData.m_hitCount = hitCount;
    perRayData.m_energy = glm::vec3(0.0f);

    switch (sbtData.m_materialType) {
    case MaterialType::Default: {
      static_cast<DefaultMaterial *>(sbtData.m_material)
          ->ApplyNormalTexture(normal, texCoord, tangent);
      float metallic =
          static_cast<DefaultMaterial *>(sbtData.m_material)->m_metallic;
      float roughness =
          static_cast<DefaultMaterial *>(sbtData.m_material)->m_roughness;
      glm::vec3 albedoColor = static_cast<DefaultMaterial *>(sbtData.m_material)
                                  ->GetAlbedo(texCoord);
      if (perRayData.m_hitCount <=
          defaultRenderingLaunchParams.m_defaultRenderingProperties
              .m_bounceLimit) {
        energy = glm::vec3(0.0f);
        float f = 1.0f;
        if (metallic >= 0.0f)
          f = (metallic + 2) / (metallic + 1);
        float3 incidentRayOrigin;
        float3 newRayDirectionInternal;
        BRDF(metallic, perRayData.m_random, normal, hitPoint, rayDirection,
             incidentRayOrigin, newRayDirectionInternal);
        optixTrace(
            defaultRenderingLaunchParams.m_traversable, incidentRayOrigin,
            newRayDirectionInternal,
            1e-3f, // tmin
            1e20f, // tmax
            0.0f,  // rayTime
            static_cast<OptixVisibilityMask>(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            static_cast<int>(
                DefaultRenderingRayType::RadianceRayType), // SBT offset
            static_cast<int>(
                DefaultRenderingRayType::RayTypeCount), // SBT stride
            static_cast<int>(
                DefaultRenderingRayType::RadianceRayType), // missSBTIndex
            u0, u1);
        energy +=
            albedoColor *
            glm::clamp(glm::abs(glm::dot(
                           normal, glm::vec3(newRayDirectionInternal.x,
                                             newRayDirectionInternal.y,
                                             newRayDirectionInternal.z))) *
                               roughness +
                           (1.0f - roughness) * f,
                       0.0f, 1.0f) *
            perRayData.m_energy;
      }
      if (hitCount == 1) {
        perRayData.m_pixelNormal = normal;
        perRayData.m_pixelAlbedo = albedoColor;
      }
      perRayData.m_energy =
          energy + static_cast<DefaultMaterial *>(sbtData.m_material)
                           ->m_diffuseIntensity *
                       albedoColor;

    } break;
    case MaterialType::MLVQ: {
      glm::vec3 btfColor;
      if (perRayData.m_hitCount <=
          defaultRenderingLaunchParams.m_defaultRenderingProperties
              .m_bounceLimit) {
        energy = glm::vec3(0.0f);
        float f = 1.0f;
        glm::vec3 reflected = Reflect(rayDirection, normal);
        glm::vec3 newRayDirection =
            RandomSampleHemisphere(perRayData.m_random, reflected, 1.0f);
        static_cast<MLVQMaterial *>(sbtData.m_material)
            ->GetValue(texCoord, rayDirection, newRayDirection, normal, tangent,
                       btfColor,
                       false /*(perRayData.m_printInfo && sampleID == 0)*/);
        auto origin = hitPoint;
        origin += normal * 1e-3f;
        float3 incidentRayOrigin = make_float3(origin.x, origin.y, origin.z);
        float3 newRayDirectionInternal = make_float3(
            newRayDirection.x, newRayDirection.y, newRayDirection.z);
        optixTrace(
            defaultRenderingLaunchParams.m_traversable, incidentRayOrigin,
            newRayDirectionInternal,
            1e-3f, // tmin
            1e20f, // tmax
            0.0f,  // rayTime
            static_cast<OptixVisibilityMask>(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
            static_cast<int>(
                DefaultRenderingRayType::RadianceRayType), // SBT offset
            static_cast<int>(
                DefaultRenderingRayType::RayTypeCount), // SBT stride
            static_cast<int>(
                DefaultRenderingRayType::RadianceRayType), // missSBTIndex
            u0, u1);
        energy += btfColor * perRayData.m_energy;
      }
      if (hitCount == 1) {
        perRayData.m_pixelNormal = normal;
        perRayData.m_pixelAlbedo = btfColor;
      }
      perRayData.m_energy = energy;
    } break;
    }
  }
}
extern "C" __global__ void __closesthit__sampleSp() {
  DefaultRenderingSpSamplerRayData &perRayData =
      *GetRayDataPointer<DefaultRenderingSpSamplerRayData>();
  assert(perRayData.m_instanceId == optixGetInstanceId());
  const auto &sbtData = *(const DefaultSbtData *)optixGetSbtDataPointer();
  const auto indices = sbtData.m_mesh.GetIndices(optixGetPrimitiveIndex());
  const auto centrics = optixGetTriangleBarycentrics();
  perRayData.m_p1 = sbtData.m_mesh.GetPosition(centrics, indices);
  perRayData.m_n1 = sbtData.m_mesh.GetNormal(centrics, indices);
  perRayData.m_found = true;
}
#pragma endregion
#pragma region Any hit functions
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __anyhit__sampleSp() {
  DefaultRenderingSpSamplerRayData &perRayData =
      *GetRayDataPointer<DefaultRenderingSpSamplerRayData>();
  if (perRayData.m_instanceId != optixGetInstanceId())
    optixIgnoreIntersection();
  const auto &sbtData = *(const DefaultSbtData *)optixGetSbtDataPointer();
  const auto indices = sbtData.m_mesh.GetIndices(optixGetPrimitiveIndex());
  const auto hitPoint =
      sbtData.m_mesh.GetPosition(optixGetTriangleBarycentrics(), indices);
  const auto origin = optixGetWorldRayOrigin();
  const float distance =
      glm::distance(hitPoint, glm::vec3(origin.x, origin.y, origin.z));
  // if (distance > sbtData.m_material.GetRadiusMax())
  // optixIgnoreIntersection();
}
#pragma endregion
#pragma region Miss functions
extern "C" __global__ void __miss__radiance() {
  DefaultRenderingRadianceRayData &perRayData =
      *GetRayDataPointer<DefaultRenderingRadianceRayData>();
  const float3 rayDir = optixGetWorldRayDirection();
  if (defaultRenderingLaunchParams.m_defaultRenderingProperties
          .m_enableGround &&
      rayDir.y < 0) {
    unsigned hitCount = perRayData.m_hitCount + 1;
    // start with some ambient term
    auto energy = glm::vec3(0.0f);
    uint32_t u0, u1;
    PackRayDataPointer(&perRayData, u0, u1);
    perRayData.m_hitCount = hitCount;
    perRayData.m_energy = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0, 1, 0);
    float metallic = defaultRenderingLaunchParams.m_defaultRenderingProperties
                         .m_groundMetallic;
    metallic = (metallic == 1.0f ? -1.0f : 1.0f / glm::pow(1.0f - metallic, 3.0f));
    float roughness = defaultRenderingLaunchParams.m_defaultRenderingProperties
                          .m_groundRoughness;
    glm::vec3 albedoColor =
        defaultRenderingLaunchParams.m_defaultRenderingProperties.m_groundColor;
    if (perRayData.m_hitCount <=
        defaultRenderingLaunchParams.m_defaultRenderingProperties
            .m_bounceLimit) {
      energy = glm::vec3(0.0f);
      float f = 1.0f;
      if (metallic >= 0.0f)
        f = (metallic + 2) / (metallic + 1);
      float3 rayOrigin = optixGetWorldRayOrigin();
      glm::vec3 rayOrig = glm::vec3(rayOrigin.x, rayOrigin.y, rayOrigin.z);
      glm::vec3 rayDirection = glm::vec3(rayDir.x, rayDir.y, rayDir.z);
      glm::vec3 hitPoint =
          rayOrig -
          rayDirection * ((rayOrig.y - defaultRenderingLaunchParams.m_defaultRenderingProperties.m_groundHeight) / rayDirection.y);
      float3 incidentRayOrigin;
      float3 newRayDirectionInternal;
      BRDF(metallic, perRayData.m_random, normal, hitPoint, rayDirection,
           incidentRayOrigin, newRayDirectionInternal);
      optixTrace(
          defaultRenderingLaunchParams.m_traversable, incidentRayOrigin,
          newRayDirectionInternal,
          1e-3f, // tmin
          1e20f, // tmax
          0.0f,  // rayTime
          static_cast<OptixVisibilityMask>(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          static_cast<int>(
              DefaultRenderingRayType::RadianceRayType),           // SBT offset
          static_cast<int>(DefaultRenderingRayType::RayTypeCount), // SBT stride
          static_cast<int>(
              DefaultRenderingRayType::RadianceRayType), // missSBTIndex
          u0, u1);
      energy +=
          albedoColor *
          glm::clamp(
              glm::abs(glm::dot(normal, glm::vec3(newRayDirectionInternal.x,
                                                  newRayDirectionInternal.y,
                                                  newRayDirectionInternal.z))) *
                      roughness +
                  (1.0f - roughness) * f,
              0.0f, 1.0f) *
          perRayData.m_energy;
    }
    if (hitCount == 1) {
      perRayData.m_pixelNormal = normal;
      perRayData.m_pixelAlbedo = albedoColor;
    }
    perRayData.m_energy = energy + 0.0f * albedoColor;
  } else {
    const glm::vec3 sunColor = glm::normalize(
        defaultRenderingLaunchParams.m_defaultRenderingProperties.m_sunColor);
    float3 environmentalLightColor =
        make_float3(sunColor.x, sunColor.y, sunColor.z);
    switch (defaultRenderingLaunchParams.m_defaultRenderingProperties
                .m_environmentalLightingType) {
    case EnvironmentalLightingType::White:
      break;
    case EnvironmentalLightingType::EnvironmentalMap:
      if (defaultRenderingLaunchParams.m_defaultRenderingProperties
              .m_environmentalMapId != 0) {
        float4 color = SampleCubeMap<float4>(
            defaultRenderingLaunchParams.m_skylight.m_environmentalMaps,
            rayDir);
        environmentalLightColor = make_float3(color.x, color.y, color.z);
      }
      break;
    case EnvironmentalLightingType::CIE:
      float skylightIntensity = CIESkyIntensity(
          glm::vec3(rayDir.x, rayDir.y, rayDir.z),
          glm::normalize(defaultRenderingLaunchParams
                             .m_defaultRenderingProperties.m_sunDirection),
          glm::vec3(0, 1, 0));
      environmentalLightColor.x *= skylightIntensity;
      environmentalLightColor.y *= skylightIntensity;
      environmentalLightColor.z *= skylightIntensity;
      break;
    }
    perRayData.m_pixelAlbedo = perRayData.m_energy =
        glm::vec3(environmentalLightColor.x, environmentalLightColor.y,
                  environmentalLightColor.z);
    perRayData.m_energy *= defaultRenderingLaunchParams.m_defaultRenderingProperties
                        .m_skylightIntensity;
  }
}
extern "C" __global__ void __miss__sampleSp() {}
#pragma endregion
#pragma region Main ray generation
extern "C" __global__ void __raygen__renderFrame() {
  float ix = optixGetLaunchIndex().x;
  float iy = optixGetLaunchIndex().y;
  const uint32_t fbIndex =
      ix + iy * defaultRenderingLaunchParams.m_frame.size.x;
  if (defaultRenderingLaunchParams.m_defaultRenderingProperties
          .m_enableGround &&
      defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera.m_from
              .y < defaultRenderingLaunchParams.m_defaultRenderingProperties
                       .m_groundHeight) {
    defaultRenderingLaunchParams.m_frame.m_colorBuffer[fbIndex] =
        glm::vec4(0.0, 0.0, 0.0, 1.0);
    defaultRenderingLaunchParams.m_frame.m_albedoBuffer[fbIndex] =
        glm::vec4(0.0, 0.0, 0.0, 1.0);
    defaultRenderingLaunchParams.m_frame.m_normalBuffer[fbIndex] =
        glm::vec4(0.0, 0.0, -1.0, 1.0);
  } else {
    // compute a test pattern based on pixel ID

    DefaultRenderingRadianceRayData cameraRayData;
    cameraRayData.m_hitCount = 0;
    cameraRayData.m_random.Init(
        ix + defaultRenderingLaunchParams.m_defaultRenderingProperties
                     .m_frameSize.x *
                 iy,
        defaultRenderingLaunchParams.m_frame.m_frameId);
    cameraRayData.m_energy = glm::vec3(0);
    cameraRayData.m_pixelNormal = glm::vec3(0);
    cameraRayData.m_pixelAlbedo = glm::vec3(0);
    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    PackRayDataPointer(&cameraRayData, u0, u1);

    const auto numPixelSamples =
        defaultRenderingLaunchParams.m_defaultRenderingProperties
            .m_samplesPerPixel;
    auto pixelColor = glm::vec3(0.f);
    auto pixelNormal = glm::vec3(0.f);
    auto pixelAlbedo = glm::vec3(0.f);

    for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
      // normalized screen plane position, in [0,1]^2
      // iw: note for de-noising that's not actually correct - if we
      // assume that the camera should only(!) cover the de-noised
      // screen then the actual screen plane we should be using during
      // rendering is slightly larger than [0,1]^2
      glm::vec2 screen;
      screen = glm::vec2(ix + cameraRayData.m_random(),
                         iy + cameraRayData.m_random()) /
               glm::vec2(defaultRenderingLaunchParams
                             .m_defaultRenderingProperties.m_frameSize);
      glm::vec3 rayDir = glm::normalize(
          defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera
              .m_direction +
          (screen.x - 0.5f) *
              defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera
                  .m_horizontal +
          (screen.y - 0.5f) *
              defaultRenderingLaunchParams.m_defaultRenderingProperties.m_camera
                  .m_vertical);
      float3 rayOrigin =
          make_float3(defaultRenderingLaunchParams.m_defaultRenderingProperties
                          .m_camera.m_from.x,
                      defaultRenderingLaunchParams.m_defaultRenderingProperties
                          .m_camera.m_from.y,
                      defaultRenderingLaunchParams.m_defaultRenderingProperties
                          .m_camera.m_from.z);
      float3 rayDirection = make_float3(rayDir.x, rayDir.y, rayDir.z);

      optixTrace(
          defaultRenderingLaunchParams.m_traversable, rayOrigin, rayDirection,
          0.f,   // tmin
          1e20f, // tmax
          0.0f,  // rayTime
          static_cast<OptixVisibilityMask>(255),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
          static_cast<int>(
              DefaultRenderingRayType::RadianceRayType),           // SBT offset
          static_cast<int>(DefaultRenderingRayType::RayTypeCount), // SBT stride
          static_cast<int>(
              DefaultRenderingRayType::RadianceRayType), // missSBTIndex
          u0, u1);
      pixelColor += cameraRayData.m_energy;
      pixelNormal += cameraRayData.m_pixelNormal;
      pixelAlbedo += cameraRayData.m_pixelAlbedo;
      cameraRayData.m_energy = glm::vec3(0.0f);
      cameraRayData.m_pixelNormal = glm::vec3(0.0f);
      cameraRayData.m_pixelAlbedo = glm::vec3(0.0f);
      cameraRayData.m_hitCount = 0;
    }
    glm::vec3 rgb(pixelColor / static_cast<float>(numPixelSamples));
    rgb = glm::pow(rgb, glm::vec3(1.0 / 2.2));


    // and write/accumulate to frame buffer ...
    if (defaultRenderingLaunchParams.m_defaultRenderingProperties
            .m_accumulate) {
      if (defaultRenderingLaunchParams.m_frame.m_frameId > 1) {
        glm::vec4 currentColor =
            defaultRenderingLaunchParams.m_frame.m_colorBuffer[fbIndex];
        glm::vec3 transferredCurrentColor = currentColor;
        rgb +=
            static_cast<float>(defaultRenderingLaunchParams.m_frame.m_frameId) *
            transferredCurrentColor;
        rgb /= static_cast<float>(
            defaultRenderingLaunchParams.m_frame.m_frameId + 1);
      }
    }
    float4 data = make_float4(rgb.r, rgb.g, rgb.b, 1.0f);
    // and write to frame buffer ...

    defaultRenderingLaunchParams.m_frame.m_colorBuffer[fbIndex] =
        glm::vec4(rgb, 1.0);
    defaultRenderingLaunchParams.m_frame.m_albedoBuffer[fbIndex] =
        glm::vec4(pixelAlbedo / static_cast<float>(numPixelSamples), 1.0f);
    defaultRenderingLaunchParams.m_frame.m_normalBuffer[fbIndex] =
        glm::vec4(pixelNormal / static_cast<float>(numPixelSamples), 1.0f);
    // surf2Dwrite(data, defaultRenderingLaunchParams.m_frame.m_outputTexture, ix
    // * sizeof(float4), iy);
  }
}
#pragma endregion
} // namespace RayTracerFacility
