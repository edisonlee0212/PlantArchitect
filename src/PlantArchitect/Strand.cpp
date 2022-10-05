//
// Created by lllll on 9/11/2022.
//

#include "Strand.hpp"
#include "DataComponents.hpp"
#include "StrandsRenderer.hpp"

using namespace PlantArchitect;
#pragma region Helpers
#pragma region Geometric

glm::ivec2 GetUpLeft(const glm::ivec2 &coordinate) {
    return {coordinate.x - 1, coordinate.y + 1};
}

glm::ivec2 GetUpRight(const glm::ivec2 &coordinate) {
    return {coordinate.x, coordinate.y + 1};
}

glm::ivec2 GetRight(const glm::ivec2 &coordinate) {
    return {coordinate.x + 1, coordinate.y};
}

glm::ivec2 GetDownRight(const glm::ivec2 &coordinate) {
    return {coordinate.x + 1, coordinate.y - 1};
}

glm::ivec2 GetDownLeft(const glm::ivec2 &coordinate) {
    return {coordinate.x, coordinate.y - 1};
}

glm::ivec2 GetLeft(const glm::ivec2 &coordinate) {
    return {coordinate.x - 1, coordinate.y};
}

bool RayLineIntersect(glm::vec2 rayOrigin, glm::vec2 rayDirection, glm::vec2 point1, glm::vec2 point2) {
    const auto v1 = rayOrigin - point1;
    const auto v2 = point2 - point1;
    const auto v3 = glm::vec2(-rayDirection.y, rayDirection.x);

    float dot = glm::dot(v2, v3);
    if (dot == 0.0f)
        return false;

    float t1 = (v2.x * v1.y - v2.y * v1.x) / dot;
    float t2 = glm::dot(v1, v3) / dot;

    //!!!!Check t2 >= 0 if we allow intersect on point 1
    if (t1 >= 0.0f && t2 > 0.0f && 1.0f - t2 >= 0.0f)
        return true;

    return false;
}

bool LineLineIntersect(glm::vec2 pa, glm::vec2 pb, glm::vec2 pc, glm::vec2 pd) {
    const auto v1 = pa - pc;
    const auto v2 = pd - pc;
    const auto v3 = glm::vec2(-(pb.y - pa.y), (pb.x - pa.x));

    float dot = glm::dot(v2, v3);
    if (dot == 0.0f)
        return false;

    float t1 = (v2.x * v1.y - v2.y * v1.x) / dot;
    float t2 = glm::dot(v1, v3) / dot;

    if (t1 > 0.0f && t1 < 1.0f && t2 > 0.0f && t2 < 1.0f)
        return true;

    return false;
}

bool InBoundary(const std::vector<glm::vec2> &boundary, const glm::vec2 &point) {
    const auto point2 = glm::vec2(1.0f, 0.0f);
    const auto point3 = glm::vec2(1.0f, 0.0f);
    int windingNumber = 0;
    const auto size = boundary.size();
    if (size < 3) return false;
    for (int i = 0; i < size - 1; i++) {
        if (RayLineIntersect(point, point2, boundary[i], boundary[i + 1]) &&
            RayLineIntersect(point, point3, boundary[i], boundary[i + 1])) {
            windingNumber++;
        }
    }
    if (RayLineIntersect(point, point2, boundary[size - 1], boundary[0]) &&
        RayLineIntersect(point, point3, boundary[size - 1], boundary[0]))
        windingNumber++;
    if (windingNumber % 2 == 1) {
        return true;
    }
    return false;
}

glm::vec2 GetPosition(const glm::ivec2 &coordinate) {
    return {coordinate.x + coordinate.y / 2.0f,
            coordinate.y * glm::cos(glm::radians(30.0f))};
}

glm::ivec2 GetCoordinate(const glm::vec2 &position) {
    int y = glm::round(position.y / glm::cos(glm::radians(30.0f)));
    return {glm::round((position.x - y / 2.0f)), y};
}

std::vector<glm::ivec2> GenerateCircle(int numOfKnots) {
    if (numOfKnots == 0) return {};
    std::vector<glm::ivec2> retVal;
    //Fill current point first
    int layer = 1;
    while (true) {
        int totalNumber = 1;
        for (int i = 0; i < layer; i++) {
            totalNumber += (i + 1) * 6;
        }
        if (totalNumber > numOfKnots) break;
        layer++;
    }
    layer--;
    retVal.emplace_back(0, 0);
    for (int i = 0; i < layer; i++) {
        glm::ivec2 walker = {-i - 1, i + 1};
        for (int side = 0; side < 6; side++) {
            for (int walk = 0; walk < i + 1; walk++) {
                switch (side) {
                    case 0:
                        walker = GetRight(walker);
                        break;
                    case 1:
                        walker = GetDownRight(walker);
                        break;
                    case 2:
                        walker = GetDownLeft(walker);
                        break;
                    case 3:
                        walker = GetLeft(walker);
                        break;
                    case 4:
                        walker = GetUpLeft(walker);
                        break;
                    case 5:
                        walker = GetUpRight(walker);
                        break;
                }
                retVal.push_back(walker);
            }
        }
    }
    //Fill the rest uniformly
    if (retVal.size() == numOfKnots) return retVal;
    int walk = 0;
    while (true) {
        for (int side = 0; side < 6; side++) {
            glm::ivec2 walker;
            switch (side) {
                case 0:
                    walker = {-layer - 1, layer + 1};
                    for (int i = 0; i < walk; i++) {
                        walker = GetRight(walker);
                    }
                    break;
                case 1:
                    walker = {0, layer + 1};
                    for (int i = 0; i < walk; i++) {
                        walker = GetDownRight(walker);
                    }
                    break;
                case 2:
                    walker = {layer + 1, 0};
                    for (int i = 0; i < walk; i++) {
                        walker = GetDownLeft(walker);
                    }
                    break;
                case 3:
                    walker = {layer + 1, -layer - 1};
                    for (int i = 0; i < walk; i++) {
                        walker = GetLeft(walker);
                    }
                    break;
                case 4:
                    walker = {0, -layer - 1};
                    for (int i = 0; i < walk; i++) {
                        walker = GetUpLeft(walker);
                    }
                    break;
                case 5:
                    walker = {-layer - 1, 0};
                    for (int i = 0; i < walk; i++) {
                        walker = GetUpRight(walker);
                    }
                    break;
            }
            retVal.push_back(walker);
            if (retVal.size() == numOfKnots) return retVal;
        }
        walk++;
    }
}

#pragma endregion
#pragma region Knot Operations

void StrandPlant::Extract(const std::shared_ptr<StrandsIntersection> &strandIntersection,
                          const std::vector<SplitSettings> &targets,
                          std::vector<std::vector<std::shared_ptr<StrandKnot>>> &extractedKnots) const {
    std::vector<bool> selected;
    selected.resize(strandIntersection->m_strandKnots.size());
    for (auto &&i: selected) i = false;
    int remainingKnotSize = strandIntersection->m_strandKnots.size();
    for (const auto &target: targets) {
        extractedKnots.emplace_back();
        auto &operatingList = extractedKnots.back();
        operatingList.clear();
        if (target.m_knotSize >= remainingKnotSize) {
            for (int knotIndex = 0; knotIndex < strandIntersection->m_strandKnots.size(); knotIndex++) {
                if (!selected[knotIndex]) operatingList.emplace_back(strandIntersection->m_strandKnots[knotIndex]);
            }
            return;
        }
        std::map<int, std::map<float, std::pair<int, std::shared_ptr<StrandKnot>>>> sortedKnots;
        for (int knotIndex = 0; knotIndex < strandIntersection->m_strandKnots.size(); knotIndex++) {
            if (selected[knotIndex]) continue;
            auto &knot = strandIntersection->m_strandKnots[knotIndex];
            auto position = GetPosition(knot->m_coordinate);
            if (position == glm::vec2(0.0f)) continue;
            int angle = glm::degrees(
                    glm::acos(glm::clamp(glm::dot(glm::normalize(position), glm::normalize(target.m_direction)), 0.0f,
                                         1.0f))) /
                        10.0f;
            float distance = glm::length(position);
            auto search = sortedKnots.find(angle);
            if (search == sortedKnots.end()) {
                sortedKnots[angle] = {{distance, std::make_pair(knotIndex, knot)}};
            } else {
                search->second[distance] = std::make_pair(knotIndex, knot);
            }
        }
        auto baseKnotPair = sortedKnots.begin()->second.rbegin()->second;
        std::map<float, std::vector<std::pair<int, std::shared_ptr<StrandKnot>>>> distanceSortedKnots;
        auto basePosition = GetPosition(baseKnotPair.second->m_coordinate);
        for (int knotIndex = 0; knotIndex < strandIntersection->m_strandKnots.size(); knotIndex++) {
            if (selected[knotIndex]) continue;
            auto &knot = strandIntersection->m_strandKnots[knotIndex];
            auto position = GetPosition(knot->m_coordinate);
            float distance = glm::distance(position, basePosition);
            auto search = distanceSortedKnots.find(distance);
            if (search == distanceSortedKnots.end()) {
                distanceSortedKnots[distance] = {std::make_pair(knotIndex, knot)};
            } else {
                search->second.emplace_back(knotIndex, knot);
            }
        }
        bool skip = false;
        for (const auto &collection: distanceSortedKnots) {
            if (skip) break;
            for (const auto &knotPair: collection.second) {
                if (operatingList.size() == target.m_knotSize) {
                    skip = true;
                    break;
                }
                operatingList.emplace_back(knotPair.second);
                selected[knotPair.first] = true;
                remainingKnotSize--;

            }
        }
    }

}

void RandomlyMatchKnots(std::vector<std::shared_ptr<StrandKnot>> &srcList,
                        std::vector<std::shared_ptr<StrandKnot>> &dstList) {
    assert(srcList.size() == dstList.size());
    auto size = srcList.size();
    for (int i = 0; i < size; i++) {
        srcList[i]->m_next = dstList[i];
        dstList[i]->m_prev = srcList[i];
    }
}

void MatchKnots(std::vector<std::shared_ptr<StrandKnot>> &srcList, std::vector<std::shared_ptr<StrandKnot>> &dstList) {
    assert(srcList.size() == dstList.size());
    auto size = srcList.size();

}

#pragma endregion
#pragma endregion

void StrandPlant::Split(const std::shared_ptr<StrandsIntersection> &strandIntersection,
                        const std::vector<SplitSettings> &targets,
                        const std::function<void(
                                std::vector<std::shared_ptr<StrandKnot>> &srcList,
                                std::vector<std::shared_ptr<StrandKnot>> &dstList)> &extendFunc) {
    strandIntersection->m_children.clear();
    std::vector<std::vector<std::shared_ptr<StrandKnot>>> splitKnots;
    Extract(strandIntersection, targets, splitKnots);
    int index = 0;
    for (auto &list: splitKnots) {
        auto child = std::make_shared<StrandsIntersection>();
        auto direction = glm::normalize(targets[index].m_direction);
        child->m_transform.SetPosition(glm::vec3(-direction.x * 0.5f, 1.0f, -direction.y * 0.5f));
        strandIntersection->m_children.push_back(child);
        child->m_parent = strandIntersection;
        child->m_strandKnots.clear();
        auto coordinates = GenerateCircle(list.size());
        for (int i = 0; i < list.size(); i++) {
            auto newKnot = std::make_shared<StrandKnot>();
            child->m_strandKnots.emplace_back(newKnot);
            newKnot->m_coordinate = coordinates[i];
        }
        extendFunc(list, child->m_strandKnots);
        for (int i = 0; i < list.size(); i++) {
            auto &knot = child->m_strandKnots[i];
            auto prev = knot->m_prev.lock();
            auto strand = prev->m_strand;
            knot->m_color = prev->m_color;
            knot->m_thickness = prev->m_thickness;
            knot->m_strand = strand;
        }
        child->CalculateConnectivity();
        index++;
    }
}

void StrandPlant::Extend(const std::shared_ptr<StrandsIntersection> &strandIntersection) {
    strandIntersection->m_children.clear();
    auto child = std::make_shared<StrandsIntersection>();
    strandIntersection->m_children.push_back(child);
    child->m_parent = strandIntersection;
    child->m_strandKnots.clear();
    for (const auto &i: strandIntersection->m_strandKnots) {
        auto newKnot = std::make_shared<StrandKnot>();
        Transform transform;
        transform.SetPosition(glm::vec3(0, 1, 0));
        child->m_transform = transform;
        child->m_strandKnots.emplace_back(newKnot);
        newKnot->m_coordinate = i->m_coordinate;
        newKnot->m_prev = i;
        newKnot->m_color = i->m_color;
        newKnot->m_thickness = i->m_thickness;
        i->m_next = newKnot;
        auto strand = i->m_strand;
        newKnot->m_strand = strand;
    }
    child->CalculateConnectivity();
}

void StrandPlant::GenerateStrands() {
    if (!m_root) return;
    m_strands.clear();
    for (const auto &knot: m_root->m_strandKnots) {
        auto strand = std::make_shared<Strand>();
        strand->m_start = knot;
        knot->m_strand = strand;
        m_strands.push_back(strand);
    }
}


void StrandPlant::OnInspect() {
    if (ImGui::Button("Generate Strands")) GenerateStrands();
    if (ImGui::Button("Initialize Renderer")) InitializeStrandRenderer();
    static bool displayEditor = true;
    ImGui::Checkbox("Display Strand Tree Editor", &displayEditor);
    if (displayEditor) {
        if (m_root) {
            if (ImGui::Begin("Strand Tree Hierarchy")) {
                bool deleted = false;
                DrawIntersectionGui(m_root, deleted, 0);
                m_selectedIntersectionHierarchyList.clear();
            }
            ImGui::End();
        }
    }


    static bool displayRegion = true;

    ImGui::Checkbox("Display Intersection", &displayRegion);
    if (displayRegion && m_root && !m_selectedStrandIntersection.expired()) {
        DisplayIntersection(m_selectedStrandIntersection.lock(), "Intersection", m_root == m_selectedStrandIntersection.lock());
    }

}

void StrandPlant::OnCreate() {
    m_root = std::make_shared<StrandsIntersection>();
    m_selectedIntersectionHierarchyList.clear();
    m_selectedStrandIntersection = m_root;
}

void StrandPlant::InitializeStrandRenderer() const {
    if (!m_root) return;
    m_root->CalculatePosition(GlobalTransform());
    auto scene = GetScene();
    auto owner = GetOwner();
    auto renderer = scene->GetOrSetPrivateComponent<StrandsRenderer>(owner).lock();
    auto strandsAsset = ProjectManager::CreateTemporaryAsset<Strands>();
    std::vector<int> strandsList;
    std::vector<StrandPoint> points;
    for (const auto &strand: m_strands) {
        strand->BuildStrands(strandsList, points);
    }
    strandsAsset->SetPoints(strandsList, points, Strands::SplineMode::Cubic);
    renderer->m_strands = strandsAsset;
    renderer->m_material = ProjectManager::CreateTemporaryAsset<Material>();
}

void StrandPlant::DrawIntersectionGui(const std::shared_ptr<StrandsIntersection> &strandIntersection, bool& deleted,
                                      const unsigned &hierarchyLevel) {
    const int index = m_selectedIntersectionHierarchyList.size() - hierarchyLevel - 1;
    if (!m_selectedIntersectionHierarchyList.empty() && index >= 0 && index < m_selectedIntersectionHierarchyList.size() &&
            m_selectedIntersectionHierarchyList[index].lock() == strandIntersection) {
        ImGui::SetNextItemOpen(true);
    }
    const bool opened = ImGui::TreeNodeEx("Intersection", ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_NoAutoOpenOnLog |
    (m_selectedStrandIntersection.lock() == strandIntersection ? ImGuiTreeNodeFlags_Framed : ImGuiTreeNodeFlags_FramePadding));
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        SetSelectedIntersection(strandIntersection, false);
    }


    deleted = DrawIntersectionMenu(strandIntersection);
    if (opened && !deleted) {
        ImGui::TreePush();
        for(int i = 0; i < strandIntersection->m_children.size(); i++){
            auto& child = strandIntersection->m_children[i];
            bool childDeleted = false;
            DrawIntersectionGui(child, childDeleted, hierarchyLevel + 1);
            if(childDeleted){
                strandIntersection->m_children.erase(strandIntersection->m_children.begin() + i);
                break;
            }
        }
        ImGui::TreePop();
    }

}

void Strand::BuildStrands(std::vector<int> &strands, std::vector<StrandPoint> &points) {
    strands.emplace_back(points.size());
    auto frontPointIndex = points.size();
    StrandPoint point;
    auto walker = m_start.lock();
    point.m_position = walker->m_position;
    point.m_thickness = walker->m_thickness;
    point.m_color = walker->m_color;
    points.emplace_back(point);
    points.emplace_back(point);

    while (!walker->m_next.expired()) {
        auto startPosition = walker->m_position;
        auto startDirection = walker->m_direction;
        auto startThickness = walker->m_thickness;
        auto startColor = walker->m_color;
        walker = walker->m_next.lock();
        auto endPosition = walker->m_position;
        auto endDirection = walker->m_direction;
        auto endThickness = walker->m_thickness;
        auto endColor = walker->m_color;
        auto distance = glm::distance(startPosition, endPosition) * 0.25f;

        point.m_position = startPosition + distance * startDirection;
        point.m_thickness = glm::mix(startThickness, endThickness, 0.25f);
        point.m_color = glm::mix(startColor, endColor, 0.25f);
        points.emplace_back(point);

        point.m_position = endPosition - distance * endDirection;
        point.m_thickness = glm::mix(startThickness, endThickness, 0.75f);
        point.m_color = glm::mix(startColor, endColor, 0.75f);
        points.emplace_back(point);

        point.m_position = endPosition;
        point.m_thickness = endThickness;
        point.m_color = endColor;
        points.emplace_back(point);
    }

    StrandPoint frontPoint;
    frontPoint = points.at(frontPointIndex);
    frontPoint.m_position = 2.0f * frontPoint.m_position - points.at(frontPointIndex + 1).m_position;
    points.at(frontPointIndex) = frontPoint;

    StrandPoint backPoint;
    backPoint = points.at(points.size() - 2);
    backPoint.m_position = 2.0f * points.at(points.size() - 1).m_position - backPoint.m_position;
    points.emplace_back(backPoint);
}

void StrandsIntersection::Construct(const std::vector<glm::vec2> &points) {
    auto copiedPoints = points;
    //1. Calculate min/max bound
    auto max = glm::vec2(FLT_MIN);
    auto min = glm::vec2(FLT_MAX);
    for (const auto &point: copiedPoints) {
        if (max.x < point.x) max.x = point.x;
        if (max.y < point.y) max.y = point.y;
        if (min.x > point.x) min.x = point.x;
        if (min.y > point.y) min.y = point.y;
    }
    auto center = (max + min) / 2.0f;
    auto boundaryRadius = (max - min) / 2.0f;
    max -= center;
    min -= center;
    for (auto &point: copiedPoints) {
        point -= center;
    }
    m_strandKnots.clear();
    auto sum = glm::ivec2(0);
    int yRange = glm::ceil(boundaryRadius.y / glm::cos(glm::radians(30.0f)));
    int xRange = glm::ceil(boundaryRadius.x);
    for (int i = -xRange; i <= xRange; i++) {
        for (int j = -yRange; j <= yRange; j++) {
            glm::ivec2 coordinate;
            coordinate.y = j;
            coordinate.x = i - j / 2;
            if (InBoundary(copiedPoints, GetPosition(coordinate))) {
                m_strandKnots.push_back(std::make_shared<StrandKnot>());
                m_strandKnots.back()->m_coordinate = coordinate;
                sum += coordinate;
            }
        }
    }
    if (m_strandKnots.empty())return;
    sum /= m_strandKnots.size();
    for (auto &knot: m_strandKnots) {
        knot->m_coordinate -= sum;
        knot->m_color = glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
    }
    CalculateConnectivity();
}

bool StrandPlant::DisplayIntersection(const std::shared_ptr<StrandsIntersection> &strandIntersection,
                                      const std::string &title, bool editable) {
    bool changed = false;
    if (ImGui::Begin(title.c_str())) {
        ImGui::Text("Knot size: %d", strandIntersection->m_strandKnots.size());
        ImGui::Text("Max distance to boundary: %d", strandIntersection->m_maxDistanceToBoundary);
        static float angle = 0.0f;
        static int numOfKnots = 200;
        bool needExtract = false;
        if (ImGui::DragFloat("Angle", &angle, 1.0f, -180.0f, 180.0f)) needExtract = true;
        if (ImGui::DragInt("Num of points", &numOfKnots, 1, 1, strandIntersection->m_strandKnots.size()))
            needExtract = true;
        if (needExtract) {
            for (auto &knot: strandIntersection->m_strandKnots) knot->m_selected = false;
            std::vector<std::vector<std::shared_ptr<StrandKnot>>> extraction;
            std::vector<SplitSettings> settings;
            settings.resize(1);
            settings[0].m_direction = glm::vec2(glm::sin(glm::radians(angle)), glm::cos(glm::radians(angle)));
            settings[0].m_knotSize = numOfKnots;
            Extract(strandIntersection, settings, extraction);
            for (auto &knot: extraction[0]) knot->m_selected = true;
        }
        if (ImGui::Button("Split")) {
            std::vector<std::vector<std::shared_ptr<StrandKnot>>> extraction;
            std::vector<SplitSettings> settings;
            settings.resize(2);
            settings[0].m_direction = glm::vec2(glm::sin(glm::radians(angle)), glm::cos(glm::radians(angle)));
            settings[0].m_knotSize = numOfKnots;
            settings[1].m_direction = glm::vec2(glm::sin(glm::radians(180.0f - angle)),
                                                glm::cos(glm::radians(180.0f - angle)));
            settings[1].m_knotSize = strandIntersection->m_strandKnots.size() - numOfKnots;
            Split(strandIntersection, settings, [&](std::vector<std::shared_ptr<StrandKnot>> &srcList,
                                                    std::vector<std::shared_ptr<StrandKnot>> &dstList) {
                RandomlyMatchKnots(srcList, dstList);
            });
        }
        static auto scrolling = glm::vec2(0.0f);
        static float zoomFactor = 10.0f;
        if (ImGui::Button("Recenter")) {
            scrolling = glm::vec2(0.0f);
        }
        ImGui::DragFloat("Zoom", &zoomFactor, zoomFactor / 100.0f, 1.0f, 50.0f);
        zoomFactor = glm::clamp(zoomFactor, 1.0f, 50.0f);
        ImGuiIO &io = ImGui::GetIO();
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
        if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
        if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
        ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
        const ImVec2 origin(canvas_p0.x + canvas_sz.x / 2.0f + scrolling.x,
                            canvas_p0.y + canvas_sz.y / 2.0f + scrolling.y); // Lock scrolled origin
        const ImVec2 mousePosInCanvas((io.MousePos.x - origin.x) / zoomFactor,
                                      (io.MousePos.y - origin.y) / zoomFactor);

        // Draw border and background color
        draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
        draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

        // This will catch our interactions
        ImGui::InvisibleButton("canvas", canvas_sz,
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        const bool isMouseHovered = ImGui::IsItemHovered(); // Hovered
        const bool isMouseActive = ImGui::IsItemActive();   // Held

        // Pan (we use a zero mouse threshold when there's no context menu)
        // You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
        const float mouse_threshold_for_pan = -1.0f;
        if (isMouseActive && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan)) {
            scrolling.x += io.MouseDelta.x;
            scrolling.y += io.MouseDelta.y;
        }
        static bool addingLine = false;
        // Context menu (under default mouse threshold)
        ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
        if (drag_delta.x == 0.0f && drag_delta.y == 0.0f)
            ImGui::OpenPopupOnItemClick("context", ImGuiPopupFlags_MouseButtonRight);
        static std::vector<glm::vec2> points = {};
        if (ImGui::BeginPopup("context")) {

            ImGui::EndPopup();
        }

        // Draw grid + all lines in the canvas
        draw_list->PushClipRect(canvas_p0, canvas_p1, true);
        if (false) {
            const float GRID_STEP = 64.0f;
            for (float x = fmodf(scrolling.x, GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
                draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y),
                                   IM_COL32(200, 200, 200, 40));
            for (float y = fmodf(scrolling.y, GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
                draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y),
                                   IM_COL32(200, 200, 200, 40));
        }
        // Add first and second point
        if (editable && isMouseHovered && !addingLine && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            points.clear();
            points.emplace_back(mousePosInCanvas.x, mousePosInCanvas.y);
            addingLine = true;
        }
        {
            for (const auto &knot: strandIntersection->m_strandKnots) {
                auto pointPosition = GetPosition(knot->m_coordinate);
                auto canvasPosition = ImVec2(origin.x + pointPosition.x * zoomFactor,
                                             origin.y + pointPosition.y * zoomFactor);
                draw_list->AddCircleFilled(canvasPosition,
                                           glm::clamp(0.4f * zoomFactor, 1.0f, 100.0f),
                                           IM_COL32(255.0f * knot->m_distanceToBoundary /
                                                    strandIntersection->m_maxDistanceToBoundary,
                                                    255.0f * knot->m_distanceToBoundary /
                                                    strandIntersection->m_maxDistanceToBoundary,
                                                    255.0f * knot->m_distanceToBoundary /
                                                    strandIntersection->m_maxDistanceToBoundary,
                                                    255));
                if (knot->m_selected) {
                    draw_list->AddCircle(canvasPosition,
                                         glm::clamp(0.5f * zoomFactor, 1.0f, 100.0f),
                                         IM_COL32(255,
                                                  255,
                                                  0, 128));
                }

                if (zoomFactor > 20) {
                    auto textCanvasPosition = ImVec2(origin.x + pointPosition.x * zoomFactor - 0.3f * zoomFactor,
                                                     origin.y + pointPosition.y * zoomFactor - 0.3f * zoomFactor);
                    auto text = std::to_string(knot->m_distanceToBoundary);
                    draw_list->AddText(nullptr, 0.5f * zoomFactor, textCanvasPosition, IM_COL32(255, 0, 0, 255),
                                       text.c_str());
                }
            }
            draw_list->AddCircle(origin,
                                 glm::clamp(0.5f * zoomFactor, 1.0f, 100.0f),
                                 IM_COL32(255,
                                          0,
                                          0, 255));
        }
        if (addingLine) {
            const auto size = points.size();
            for (int i = 0; i < size - 1; i++) {
                draw_list->AddLine(ImVec2(origin.x + points[i].x * zoomFactor,
                                          origin.y + points[i].y * zoomFactor),
                                   ImVec2(origin.x + points[i + 1].x * zoomFactor,
                                          origin.y + points[i + 1].y * zoomFactor),
                                   IM_COL32(255, 0, 0, 255), 2.0f);
            }
            if (glm::distance(points.back(), {mousePosInCanvas.x, mousePosInCanvas.y}) >= 10.0f / zoomFactor)
                points.emplace_back(mousePosInCanvas.x, mousePosInCanvas.y);
            if (editable && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                addingLine = false;
                if (!strandIntersection->CheckBoundary(points)) {
                    strandIntersection->Construct(points);
                    changed = true;
                }
            }
        }
        draw_list->PopClipRect();
    }
    ImGui::End();
    return changed;
}

void StrandPlant::SetSelectedIntersection(const std::shared_ptr<StrandsIntersection>& strandIntersection, bool openMenu) {
    if (strandIntersection == m_selectedStrandIntersection.lock())
        return;
    m_selectedIntersectionHierarchyList.clear();
    if (!strandIntersection) {
        m_selectedStrandIntersection.reset();
        return;
    }
    auto scene = GetScene();
    m_selectedStrandIntersection = strandIntersection;
    if (!openMenu)
        return;
    auto walker = strandIntersection;
    while (!walker->m_parent.expired()) {
        m_selectedIntersectionHierarchyList.push_back(walker);
        walker = walker->m_parent.lock();
    }
}

bool StrandPlant::DrawIntersectionMenu(const std::shared_ptr<StrandsIntersection> &strandIntersection) {
    bool deleted = false;
    if (ImGui::BeginPopupContextItem(std::to_string(strandIntersection->m_handle).c_str())) {
        auto scene = GetScene();
        ImGui::Text(("Handle: " + std::to_string(strandIntersection->m_handle.GetValue())).c_str());
        if (ImGui::Button("Delete")) {
            deleted = true;
        }
        const std::string tag = "##StrandIntersection" + std::to_string(strandIntersection->m_handle);
        if (!deleted && ImGui::BeginMenu(("Rename" + tag).c_str())) {
            static char newName[256];
            ImGui::InputText("New name", newName, 256);
            if (ImGui::Button("Confirm")) {
                strandIntersection->m_name = newName;
                memset(newName, 0, 256);
            }
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }
    return deleted;
}

void StrandsIntersection::CalculateConnectivity() {
    m_maxDistanceToBoundary = 0;
    std::map<std::pair<int, int>, std::shared_ptr<StrandKnot>> knotsMap;
    for (const auto &knot: m_strandKnots) {
        knot->m_distanceToBoundary = 99999;
        knotsMap[{knot->m_coordinate.x, knot->m_coordinate.y}] = knot;
    }

    std::queue<std::shared_ptr<StrandKnot>> knotDistanceCalculationQueue;
    std::set<std::shared_ptr<StrandKnot>> visitedKnots;
    for (auto &knot: m_strandKnots) {
        int x = knot->m_coordinate.x;
        int y = knot->m_coordinate.y;
        knot->m_upLeft.reset();
        knot->m_upRight.reset();
        knot->m_right.reset();
        knot->m_downRight.reset();
        knot->m_downLeft.reset();
        knot->m_left.reset();
        auto search = knotsMap.find({x - 1, y + 1});
        if (search != knotsMap.end()) {
            knot->m_upLeft = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x, y + 1});
        if (search != knotsMap.end()) {
            knot->m_upRight = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x + 1, y});
        if (search != knotsMap.end()) {
            knot->m_right = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x + 1, y - 1});
        if (search != knotsMap.end()) {
            knot->m_downRight = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x, y - 1});
        if (search != knotsMap.end()) {
            knot->m_downLeft = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
        search = knotsMap.find({x - 1, y});
        if (search != knotsMap.end()) {
            knot->m_left = search->second;
        } else {
            knot->m_distanceToBoundary = 0;
            knotDistanceCalculationQueue.push(knot);
            visitedKnots.emplace(knot);
        }
    }

    while (!knotDistanceCalculationQueue.empty()) {
        auto currentKnot = knotDistanceCalculationQueue.front();
        knotDistanceCalculationQueue.pop();
        if (!currentKnot->m_upLeft.expired()) {
            auto visitingKnot = currentKnot->m_upLeft.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_upRight.expired()) {
            auto visitingKnot = currentKnot->m_upRight.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_right.expired()) {
            auto visitingKnot = currentKnot->m_right.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_downRight.expired()) {
            auto visitingKnot = currentKnot->m_downRight.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_downLeft.expired()) {
            auto visitingKnot = currentKnot->m_downLeft.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (!currentKnot->m_left.expired()) {
            auto visitingKnot = currentKnot->m_left.lock();
            if (currentKnot->m_distanceToBoundary + 1 < visitingKnot->m_distanceToBoundary) {
                visitingKnot->m_distanceToBoundary = currentKnot->m_distanceToBoundary + 1;
            }
            if (visitedKnots.find(visitingKnot) == visitedKnots.end()) {
                visitedKnots.emplace(visitingKnot);
                knotDistanceCalculationQueue.push(visitingKnot);
            }
        }
        if (currentKnot->m_distanceToBoundary > m_maxDistanceToBoundary)
            m_maxDistanceToBoundary = currentKnot->m_distanceToBoundary;
    }
}

std::vector<std::shared_ptr<StrandKnot>> StrandsIntersection::GetBoundaryKnots() const {
    std::vector<std::shared_ptr<StrandKnot>> retVal;
    for (const auto &knot: m_strandKnots) {
        if (knot->m_distanceToBoundary == 0) {
            auto walker = knot;
            if (!knot->m_upLeft.expired() && knot->m_upLeft.lock()->m_distanceToBoundary == 0)
                walker = knot->m_upLeft.lock();
            else if (!knot->m_upRight.expired() && knot->m_upRight.lock()->m_distanceToBoundary == 0)
                walker = knot->m_upRight.lock();
            else if (!knot->m_right.expired() && knot->m_right.lock()->m_distanceToBoundary == 0)
                walker = knot->m_right.lock();
            else if (!knot->m_downRight.expired() && knot->m_downRight.lock()->m_distanceToBoundary == 0)
                walker = knot->m_downRight.lock();
            else if (!knot->m_downLeft.expired() && knot->m_downLeft.lock()->m_distanceToBoundary == 0)
                walker = knot->m_downLeft.lock();
            else if (!knot->m_left.expired() && knot->m_left.lock()->m_distanceToBoundary == 0)
                walker = knot->m_left.lock();
            auto prev = knot;
            while (walker != knot) {
                retVal.emplace_back(walker);
                auto save = walker;
                if (!walker->m_upLeft.expired() && walker->m_upLeft.lock() != prev &&
                    walker->m_upLeft.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_upLeft.lock();
                else if (!walker->m_upRight.expired() && walker->m_upRight.lock() != prev &&
                         walker->m_upRight.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_upRight.lock();
                else if (!walker->m_right.expired() && walker->m_right.lock() != prev &&
                         walker->m_right.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_right.lock();
                else if (!walker->m_downRight.expired() && walker->m_downRight.lock() != prev &&
                         walker->m_downRight.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_downRight.lock();
                else if (!walker->m_downLeft.expired() && walker->m_downLeft.lock() != prev &&
                         walker->m_downLeft.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_downLeft.lock();
                else if (!walker->m_left.expired() && walker->m_left.lock() != prev &&
                         walker->m_left.lock()->m_distanceToBoundary == 0)
                    walker = walker->m_left.lock();
                prev = save;
            }
            retVal.emplace_back(walker);
            return retVal;
        }
    }
}


bool StrandsIntersection::CheckBoundary(const std::vector<glm::vec2> &points) {
    for (int i = 0; i < points.size(); i++) {
        auto &pa = points[(i == 0 ? points.size() - 1 : i - 1)];
        auto &pb = points[i];
        for (int j = 0; j < points.size(); j++) {
            auto &pc = points[(j == 0 ? points.size() - 1 : j - 1)];
            auto &pd = points[j];
            if (LineLineIntersect(pa, pb, pc, pd)) return true;
        }
    }
    return false;
}

void StrandsIntersection::CalculatePosition(const GlobalTransform &parentGlobalTransform) const {
    GlobalTransform globalTransform;
    globalTransform.m_value = parentGlobalTransform.m_value * m_transform.m_value;

    auto rotation = globalTransform.GetRotation();
    auto position = globalTransform.GetPosition();

    auto front = rotation * glm::vec3(0, 0, -1);
    auto up = rotation * glm::vec3(0, 1, 0);
    auto left = rotation * glm::vec3(1, 0, 0);

    for (const auto &knot: m_strandKnots) {
        auto surfacePosition = GetPosition(knot->m_coordinate);
        knot->m_position =
                position + front * surfacePosition.y * m_unitDistance + left * surfacePosition.x * m_unitDistance;
        knot->m_direction = up;
        knot->m_thickness = m_unitDistance / 3.0f;
    }

    for (const auto &i: m_children) {
        i->CalculatePosition(globalTransform);
    }
}


