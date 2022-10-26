#pragma once

#include "plant_architect_export.h"

#define InternodeHandle int
#define BranchHandle int
using namespace UniEngine;
namespace Orchards {
    /*
    struct InternodeHandle {
        int m_value;
    };
    struct BranchHandle {
        int m_value;
    };
    */

    enum class PLANT_ARCHITECT_API InternodeStatus{
        Elongating,
        Grown,
    };

    template<typename InternodeData>
    class Internode {
    public:
        InternodeData m_data;
        bool m_endNode = true;
        bool m_recycled = false;
        InternodeHandle m_handle = -1;
        BranchHandle m_branchHandle = -1;

        InternodeHandle m_parent = -1;
        std::vector<InternodeHandle> m_children;


        glm::vec3 m_globalPosition = glm::vec3(0.0f);
        glm::quat m_globalRotation = glm::vec3(0.0f);

        explicit Internode(InternodeHandle handle);

        float m_length = 0.0f;
        float m_thickness = 0.1f;
        glm::quat m_localRotation = glm::vec3(0.0f);
    };


    template<typename BranchData>
    class Branch {
    public:
        BranchData m_data;
        bool m_recycled = false;
        BranchHandle m_handle = -1;
        std::vector<InternodeHandle> m_internodes;

        BranchHandle m_parent = -1;
        std::vector<BranchHandle> m_children;

        glm::vec3 m_globalStartPosition = glm::vec3(0.0f);
        glm::quat m_globalStartRotation = glm::vec3(0.0f);
        float m_startThickness = 0.0f;

        glm::vec3 m_globalEndPosition = glm::vec3(0.0f);
        glm::quat m_globalEndRotation = glm::vec3(0.0f);
        float m_endThickness = 0.0f;

        explicit Branch(BranchHandle handle);
    };


    template<typename BranchData, typename InternodeData>
    class Plant {
        std::vector<Branch<BranchData>> m_branches;
        std::vector<Internode<InternodeData>> m_internodes;
        std::queue<InternodeHandle> m_internodePool;
        std::queue<BranchHandle> m_branchPool;

        int m_newVersion = 0;
        int m_version = -1;
        std::vector<InternodeHandle> m_sortedInternodeList;
        std::vector<BranchHandle> m_sortedBranchList;

        InternodeHandle AllocateInternode();

        void RecycleInternodeSingle(InternodeHandle handle);

        void RecycleBranchSingle(BranchHandle handle);

        BranchHandle AllocateBranch();

        void SetParentBranch(BranchHandle targetHandle, BranchHandle parentHandle);

        void DetachChildBranch(BranchHandle targetHandle, BranchHandle childHandle);

        void SetParentInternode(InternodeHandle targetHandle, InternodeHandle parentHandle);

        void DetachChildInternode(InternodeHandle targetHandle, InternodeHandle childHandle);

    public:
        void PruneInternode(InternodeHandle handle);

        void PruneBranch(BranchHandle handle);

        InternodeHandle Extend(InternodeHandle targetHandle, bool createNewBranch);

        const std::vector<InternodeHandle>& GetSortedInternodeList();

        const std::vector<BranchHandle>& GetSortedBranchList();

        void SortLists();

        Plant();

        int GetVersion() const;

        void CalculateBranches();

        Internode<InternodeData> &RefInternode(InternodeHandle handle);
        Branch<BranchData> &RefBranch(BranchHandle handle);
    };

#pragma region Helper
    template<typename BranchData, typename InternodeData>
    Branch<BranchData> &Plant<BranchData, InternodeData>::RefBranch(int handle) {
        assert(handle >= 0 && handle < m_branches.size());
        return m_branches[handle];
    }

    template<typename BranchData, typename InternodeData>
    Internode<InternodeData> &Plant<BranchData, InternodeData>::RefInternode(int handle) {
        assert(handle >= 0 && handle < m_internodes.size());
        return m_internodes[handle];
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::SortLists() {
        if(m_version == m_newVersion) return;
        m_version = m_newVersion;
        m_sortedBranchList.clear();
        std::queue<BranchHandle> branchWaitList;
        branchWaitList.push(0);
        while (!branchWaitList.empty()) {
            m_sortedBranchList.emplace_back(branchWaitList.front());
            branchWaitList.pop();
            for (const auto &i: m_branches[m_sortedBranchList.back()].m_children) {
                branchWaitList.push(i);
            }
        }

        m_sortedInternodeList.clear();
        std::queue<InternodeHandle> internodeWaitList;
        internodeWaitList.push(0);
        while (!internodeWaitList.empty()) {
            m_sortedInternodeList.emplace_back(internodeWaitList.front());
            internodeWaitList.pop();
            for (const auto &i: m_internodes[m_sortedInternodeList.back()].m_children) {
                internodeWaitList.push(i);
            }
        }
    }

    template<typename BranchData, typename InternodeData>
    const std::vector<BranchHandle>& Plant<BranchData, InternodeData>::GetSortedBranchList() {
        SortLists();
        return m_sortedBranchList;
    }

    template<typename BranchData, typename InternodeData>
    const std::vector<InternodeHandle>& Plant<BranchData, InternodeData>::GetSortedInternodeList() {
        SortLists();
        return m_sortedInternodeList;
    }

    template<typename BranchData, typename InternodeData>
    InternodeHandle Plant<BranchData, InternodeData>::Extend(int targetHandle, bool createNewBranch) {
        assert(targetHandle < m_internodes.size());
        auto &targetInternode = m_internodes[targetHandle];
        assert(!targetInternode.m_recycled);
        assert(targetInternode.m_branchHandle < m_branches.size());
        auto &branch = m_branches[targetInternode.m_branchHandle];
        assert(!branch.m_recycled);
        auto newInternodeHandle = AllocateInternode();
        SetParentInternode(newInternodeHandle, targetHandle);
        auto& originalInternode = m_internodes[targetHandle];
        auto &newInternode = m_internodes[newInternodeHandle];

        if (createNewBranch) {
            auto newBranchHandle = AllocateBranch();
            auto &newBranch = m_branches[newBranchHandle];
            SetParentBranch(newBranchHandle, originalInternode.m_branchHandle);
            newInternode.m_branchHandle = newBranchHandle;
            newBranch.m_internodes.emplace_back(newInternodeHandle);
        } else {
            originalInternode.m_endNode = false;
            branch.m_internodes.emplace_back(newInternodeHandle);
            newInternode.m_branchHandle = originalInternode.m_branchHandle;


        }
        m_newVersion++;
        return newInternodeHandle;
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::PruneBranch(int handle) {
        assert(handle != 0);
        assert(!m_branches[handle].m_recycled);
        auto &branch = m_branches[handle];
        //Remove children
        auto children = branch.m_children;
        for (const auto &child: children) {
            PruneBranch(child);
        }
        //Detach from parent
        if (branch.m_parent != -1) DetachChildBranch(branch.m_parent, handle);
        //Remove internodes
        if (!branch.m_internodes.empty()) {
            //Detach first internode from parent.
            auto internodes = branch.m_internodes;
            for (const auto &i: internodes) {
                RecycleInternodeSingle(i);
            }
        }
        RecycleBranchSingle(handle);
        m_newVersion++;
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::PruneInternode(int handle) {
        assert(handle != 0);
        assert(!m_internodes[handle].m_recycled);
        auto &internode = m_internodes[handle];
        auto branchHandle = internode.m_branchHandle;
        auto &branch = m_branches[branchHandle];
        if (handle == branch.m_internodes[0]) {
            PruneBranch(internode.m_branchHandle);
            return;
        }
        //Collect list of subsequent internodes
        std::vector<InternodeHandle> subsequentInternodes;
        while(branch.m_internodes.back() != handle){
            subsequentInternodes.emplace_back(branch.m_internodes.back());
            branch.m_internodes.pop_back();
        }
        subsequentInternodes.emplace_back(branch.m_internodes.back());
        branch.m_internodes.pop_back();
        assert(!branch.m_internodes.empty());
        //Detach from parent
        if (internode.m_parent != -1) DetachChildInternode(internode.m_parent, handle);
        //From end node remove one by one.
        InternodeHandle prev = -1;
        for (const auto &i: subsequentInternodes) {
            auto children = m_internodes[i].m_children;
            for (const auto &childInternodeHandle: children) {
                if(childInternodeHandle == prev) continue;
                auto& child = m_internodes[childInternodeHandle];
                assert(!child.m_recycled);
                auto childBranchHandle = child.m_branchHandle;
                if (childBranchHandle != branchHandle) {
                    PruneBranch(childBranchHandle);
                }
            }
            prev = i;
            RecycleInternodeSingle(i);

        }
        m_newVersion++;
    }

#pragma endregion
#pragma region Internal

    template<typename InternodeData>
    Internode<InternodeData>::Internode(int handle) {
        m_handle = handle;
        m_recycled = false;
        m_endNode = true;
        m_data = {};
    }

    template<typename BranchData>
    Branch<BranchData>::Branch(int handle) {
        m_handle = handle;
        m_recycled = false;
        m_data = {};
    }

    template<typename BranchData, typename InternodeData>
    Plant<BranchData, InternodeData>::Plant() {
        AllocateBranch();
        AllocateInternode();
        auto &rootBranch = m_branches[0];
        auto &rootInternode = m_internodes[0];
        rootInternode.m_branchHandle = 0;
        rootBranch.m_internodes.emplace_back(0);
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::DetachChildInternode(int targetHandle, int childHandle) {
        assert(targetHandle >= 0 && childHandle >= 0 && targetHandle < m_internodes.size() &&
               childHandle < m_internodes.size());
        auto &targetInternode = m_internodes[targetHandle];
        auto &childInternode = m_internodes[childHandle];
        assert(!targetInternode.m_recycled);
        assert(!childInternode.m_recycled);
        auto &children = targetInternode.m_children;
        for (int i = 0; i < children.size(); i++) {
            if (children[i] == childHandle) {
                children[i] = children.back();
                children.pop_back();
                childInternode.m_parent = -1;
                return;
            }
        }
        UNIENGINE_ERROR("Children internode not exist!")
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::SetParentInternode(int targetHandle, int parentHandle) {
        assert(targetHandle >= 0 && parentHandle >= 0 && targetHandle < m_internodes.size() &&
               parentHandle < m_internodes.size());
        auto &targetInternode = m_internodes[targetHandle];
        auto &parentInternode = m_internodes[parentHandle];
        assert(!targetInternode.m_recycled);
        assert(!parentInternode.m_recycled);
        targetInternode.m_parent = parentHandle;
#ifdef _DEBUG
        auto &children = parentInternode.m_children;
        for (int i = 0; i < children.size(); i++) {
            if (targetHandle == parentHandle) {
                UNIENGINE_ERROR("Children internode exists!")
                return;
            }
        }
#endif
        parentInternode.m_children.emplace_back(targetHandle);
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::DetachChildBranch(int targetHandle, int childHandle) {
        assert(targetHandle >= 0 && childHandle >= 0 && targetHandle < m_branches.size() &&
               childHandle < m_branches.size());
        auto &targetBranch = m_branches[targetHandle];
        auto &childBranch = m_branches[childHandle];
        assert(!targetBranch.m_recycled);
        assert(!childBranch.m_recycled);

        if(!childBranch.m_internodes.empty()){
            auto firstInternodeHandle = childBranch.m_internodes[0];
            auto &firstInternode = m_internodes[firstInternodeHandle];
            if (firstInternode.m_parent != -1)
                DetachChildInternode(firstInternode.m_parent, firstInternodeHandle);
        }

        auto &children = targetBranch.m_children;
        for (int i = 0; i < children.size(); i++) {
            if (children[i] == childHandle) {
                children[i] = children.back();
                children.pop_back();
                childBranch.m_parent = -1;
                return;
            }
        }
        UNIENGINE_ERROR("Children branch not exist!")
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::SetParentBranch(int targetHandle, int parentHandle) {
        assert(targetHandle >= 0 && parentHandle >= 0 && targetHandle < m_branches.size() &&
               parentHandle < m_branches.size());
        auto &targetBranch = m_branches[targetHandle];
        auto &parentBranch = m_branches[parentHandle];
        assert(!targetBranch.m_recycled);
        assert(!parentBranch.m_recycled);
        targetBranch.m_parent = parentHandle;
#ifdef _DEBUG
        auto &children = parentBranch.m_children;
        for (int i = 0; i < children.size(); i++) {
            if (targetHandle == parentHandle) {
                UNIENGINE_ERROR("Children branch exists!")
                return;
            }
        }
#endif
        parentBranch.m_children.emplace_back(targetHandle);
    }

    template<typename BranchData, typename InternodeData>
    BranchHandle Plant<BranchData, InternodeData>::AllocateBranch() {
        if (m_branchPool.empty()) {
            auto newBranch = m_branches.emplace_back(m_branches.size());
            return newBranch.m_handle;
        }
        auto handle = m_branchPool.front();
        m_branchPool.pop();
        auto &branch = m_branches[handle];
        branch.m_recycled = false;
        return handle;
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::RecycleBranchSingle(int handle) {
        assert(!m_branches[handle].m_recycled);
        auto &branch = m_branches[handle];
        branch.m_parent = -1;
        branch.m_children.clear();
        branch.m_internodes.clear();

        branch.m_recycled = true;
        m_branchPool.emplace(handle);
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::RecycleInternodeSingle(int handle) {
        assert(!m_internodes[handle].m_recycled);
        auto &internode = m_internodes[handle];
        internode.m_parent = -1;
        internode.m_branchHandle = -1;
        internode.m_endNode = true;
        internode.m_children.clear();

        internode.m_recycled = true;
        m_internodePool.emplace(handle);
    }

    template<typename BranchData, typename InternodeData>
    InternodeHandle Plant<BranchData, InternodeData>::AllocateInternode() {
        if (m_internodePool.empty()) {
            auto newInternode = m_internodes.emplace_back(m_internodes.size());
            return newInternode.m_handle;
        }
        auto handle = m_internodePool.front();
        m_internodePool.pop();
        auto &internode = m_internodes[handle];
        internode.m_recycled = false;
        return handle;
    }

    template<typename BranchData, typename InternodeData>
    int Plant<BranchData, InternodeData>::GetVersion() const{
        return m_version;
    }

    template<typename BranchData, typename InternodeData>
    void Plant<BranchData, InternodeData>::CalculateBranches() {
        const auto &sortedBranchList = GetSortedBranchList();
        for (const auto &branchHandle: sortedBranchList)
        {
            auto& branch = m_branches[branchHandle];
            auto& firstInternode = m_internodes.front();
            auto& lastInternode = m_internodes.back();
            branch.m_startThickness = firstInternode.m_thickness;
            branch.m_globalStartPosition = firstInternode.m_globalPosition;
            branch.m_globalStartRotation = firstInternode.m_localRotation;

            branch.m_endThickness = lastInternode.m_thickness;
            branch.m_globalEndPosition = lastInternode.m_globalPosition + lastInternode.m_length * lastInternode.m_localRotation * glm::vec3(0, 0, -1);
            branch.m_globalEndRotation = lastInternode.m_globalRotation;
        }
    }


#pragma endregion
}