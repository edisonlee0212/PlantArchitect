//
// Created by lllll on 10/22/2021.
//

#include "Camera2DVectorField.hpp"

using namespace PlantArchitect;

void Camera2DVectorField::Construct() {
    m_vectorField.clear();
    m_vectorField.resize(m_resolution.x);
    for(auto& i : m_vectorField){
        i.resize(m_resolution.y);
        for(auto j : i){
            j = glm::vec2(0, 0);
        }
    }
}
