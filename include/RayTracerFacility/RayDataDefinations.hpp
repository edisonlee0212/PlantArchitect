#pragma once

#include <Optix7.hpp>

#include <Vertex.hpp>

#include <glm/glm.hpp>

namespace RayTracerFacility {
    struct Mesh {
        glm::vec3 *m_positions;
        glm::vec3 *m_normals;
        glm::vec3 *m_tangents;
        glm::vec2 *m_texCoords;

        glm::uvec3 *m_triangles;
        glm::mat4 m_transform;

        __device__ glm::uvec3 GetIndices(const int &primitiveId) const {
            return m_triangles[primitiveId];
        }

        __device__ glm::vec2 GetTexCoord(const float2 &triangleBarycentrics,
                                         const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_texCoords[triangleIndices.x] +
                   triangleBarycentrics.x * m_texCoords[triangleIndices.y] +
                   triangleBarycentrics.y * m_texCoords[triangleIndices.z];
        }

        __device__ glm::vec3 GetPosition(const float2 &triangleBarycentrics,
                                         const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_positions[triangleIndices.x] +
                   triangleBarycentrics.x * m_positions[triangleIndices.y] +
                   triangleBarycentrics.y * m_positions[triangleIndices.z];
        }

        __device__ glm::vec3 GetNormal(const float2 &triangleBarycentrics,
                                       const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_normals[triangleIndices.x] +
                   triangleBarycentrics.x * m_normals[triangleIndices.y] +
                   triangleBarycentrics.y * m_normals[triangleIndices.z];
        }

        __device__ glm::vec3 GetTangent(const float2 &triangleBarycentrics,
                                        const glm::uvec3 &triangleIndices) const {
            return (1.f - triangleBarycentrics.x - triangleBarycentrics.y) *
                   m_tangents[triangleIndices.x] +
                   triangleBarycentrics.x * m_tangents[triangleIndices.y] +
                   triangleBarycentrics.y * m_tangents[triangleIndices.z];
        }
    };

    struct DefaultMaterial {
        glm::vec3 m_surfaceColor;
        float m_roughness = 15;
        float m_metallic = 0.5;
        cudaTextureObject_t m_albedoTexture;
        cudaTextureObject_t m_normalTexture;
        float m_diffuseIntensity;

        __device__ glm::vec3 GetAlbedo(const glm::vec2 &texCoord) const {
            if (!m_albedoTexture)
                return m_surfaceColor;
            float4 textureAlbedo =
                    tex2D<float4>(m_albedoTexture, texCoord.x, texCoord.y);
            return glm::vec3(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z);
        }

        __device__ void ApplyNormalTexture(glm::vec3 &normal,
                                           const glm::vec2 &texCoord,
                                           const glm::vec3 &tangent) const {
            if (!m_normalTexture)
                return;
            float4 textureNormal =
                    tex2D<float4>(m_normalTexture, texCoord.x, texCoord.y);
            glm::vec3 B = glm::cross(normal, tangent);
            glm::mat3 TBN = glm::mat3(tangent, B, normal);
            normal =
                    glm::vec3(textureNormal.x, textureNormal.y, textureNormal.z) * 2.0f -
                    glm::vec3(1.0f);
            normal = glm::normalize(TBN * normal);
        }

        __device__ float GetRadiusMax() const { return 0.5f; }
    };

    struct DefaultSbtData {
        Mesh m_mesh;
        DefaultMaterial m_material;
    };

/*! SBT record for a raygen program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultRenderingRayGenRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void *m_data;
    };

/*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultRenderingRayMissRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void *m_data;
    };

/*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DefaultRenderingRayHitRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        DefaultSbtData m_data;
    };

/*! SBT record for a raygen program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    DefaultIlluminationEstimationRayGenRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void *m_data;
    };

/*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    DefaultIlluminationEstimationRayMissRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        void *m_data;
    };

/*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT)
    DefaultIlluminationEstimationRayHitRecord {
        __align__(
                OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        // just a dummy value - later examples will use more interesting
        // data here
        DefaultSbtData m_data;
    };
} // namespace RayTracerFacility
