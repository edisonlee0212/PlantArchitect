//
// Created by lllll on 9/6/2021.
//

#include "IInternodeFoliage.hpp"

void PlantArchitect::IInternodeFoliage::OnInspect() {
    Editor::DragAndDropButton<Texture2D>(m_foliageTexture, "Foliage texture", true);
}

void PlantArchitect::IInternodeFoliage::Serialize(YAML::Emitter &out) {
    m_foliageTexture.Save("m_foliageTexture", out);
}

void PlantArchitect::IInternodeFoliage::Deserialize(const YAML::Node &in) {
    m_foliageTexture.Load("m_foliageTexture", in);
}

void PlantArchitect::IInternodeFoliage::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_foliageTexture);
}
