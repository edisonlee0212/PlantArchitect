//
// Created by lllll on 9/6/2021.
//

#include "DefaultInternodePhyllotaxis.hpp"
#include "Internode.hpp"
#include "InternodeSystem.hpp"
using namespace PlantArchitect;
void DefaultInternodePhyllotaxis::GenerateFoliage(const std::shared_ptr<Internode>& internode,
                                                                  const InternodeInfo &internodeInfo,
                                                                  const GlobalTransform &relativeGlobalTransform) {
    internode->m_foliageMatrices.clear();
    for(int i = 0; i < internodeInfo.m_leafCount; i++){
        const auto transform =
                relativeGlobalTransform.m_value *
                (glm::translate(glm::linearRand(glm::vec3(-internodeInfo.m_length), glm::vec3(internodeInfo.m_length)) +
                                glm::linearRand(glm::vec3(0, 0, 0), glm::vec3(0, 0, -internodeInfo.m_length))) *
                 glm::mat4_cast(glm::quat(glm::radians(
                         glm::linearRand(glm::vec3(-15.0f), glm::vec3(15.0f))))) *
                 glm::scale(glm::vec3(0.1f, 1.0f, 0.2f)));
        internode->m_foliageMatrices.push_back(transform);
    }
}
