#pragma once

#include <plant_architect_export.h>

using namespace UniEngine;
namespace PlantArchitect {
    class Internode;

    struct InternodeInfo;

    class PLANT_ARCHITECT_API InternodeFoliage : public ISerializable {

        template<typename T = IAsset>
        bool DragAndDropButton(AssetRef &target, const std::string &name,
                               const std::vector<std::string> &acceptableTypeNames, bool removable = true);

    public:
        AssetRef m_foliagePhyllotaxis;

        void Generate(const std::shared_ptr<Internode> &internode, const InternodeInfo &internodeInfo,
                      const GlobalTransform &relativeGlobalTransform);

        void OnInspect();
    };

    template<typename T>
    bool InternodeFoliage::DragAndDropButton(AssetRef &target, const std::string &name,
                                             const std::vector<std::string> &acceptableTypeNames, bool removable) {
        ImGui::Text(name.c_str());
        ImGui::SameLine();
        const std::shared_ptr<IAsset> ptr = target.Get<IAsset>();
        bool statusChanged = false;
        ImGui::Button(ptr ? ptr->m_name.c_str() : "none");
        if (ptr) {
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                EditorManager::GetInstance().m_inspectingAsset = ptr;
            }
            const std::string tag = "##" + ptr->GetTypeName() + (ptr ? std::to_string(ptr->GetHandle()) : "");
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload(ptr->GetTypeName().c_str(), &ptr, sizeof(std::shared_ptr<IAsset>));
                ImGui::TextColored(ImVec4(0, 0, 1, 1), (ptr->m_name + tag).c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginPopupContextItem(tag.c_str())) {
                if (ImGui::BeginMenu(("Rename" + tag).c_str())) {
                    static char newName[256];
                    ImGui::InputText(("New name" + tag).c_str(), newName, 256);
                    if (ImGui::Button(("Confirm" + tag).c_str()))
                        ptr->m_name = std::string(newName);
                    ImGui::EndMenu();
                }
                if (removable) {
                    if (ImGui::Button(("Remove" + tag).c_str())) {
                        target.Clear();
                        statusChanged = true;
                    }
                }
                ImGui::EndPopup();
            }
        }
        for(const auto& typeName : acceptableTypeNames) {
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload(typeName.c_str())) {
                    IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<IAsset>));
                    std::shared_ptr<T> payload_n =
                            std::dynamic_pointer_cast<T>(*static_cast<std::shared_ptr<IAsset> *>(payload->Data));
                    if (!ptr || payload_n.get() != ptr.get()) {
                        target = payload_n;
                        statusChanged = true;
                    }
                }
                ImGui::EndDragDropTarget();
            }
        }
        return statusChanged;
    }

}