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
    enum class BudStatus {
        Sleeping,
        Flushed,
        Died
    };

    template<typename BudData>
    class Bud {
    public:
        BudData m_data;
        BudStatus m_status = BudStatus::Sleeping;
        glm::quat m_localRotation = glm::vec3(0.0f);

        Bud();
    };


    template<typename InternodeData, typename BudData>
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

        /**
         * List of buds, first one will always be the apical bud which points forward.
         */
        std::vector<Bud<BudData>> m_buds;


        explicit Internode(InternodeHandle handle);

        float m_length = 0.0f;
        float m_thickness = 0.0f;
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

        GlobalTransform m_globalTransform = {};

        explicit Branch(BranchHandle handle);
    };


    template<typename BranchData, typename InternodeData, typename BudData>
    class Plant {
        std::vector<Branch<BranchData>> m_branches;
        std::vector<Internode<InternodeData, BudData>> m_internodes;
        std::queue<InternodeHandle> m_internodePool;
        std::queue<BranchHandle> m_branchPool;

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

        InternodeHandle Extend(InternodeHandle targetHandle);

        std::vector<InternodeHandle> GetSortedInternodeList();

        std::vector<BranchHandle> GetSortedBranchList();

        Plant();

        Internode<InternodeData, BudData> &RefInternode(InternodeHandle handle);
        Branch<BranchData> &RefBranch(BranchHandle handle);
    };

#pragma region Helper
    template<typename BranchData, typename InternodeData, typename BudData>
    Branch<BranchData> &Plant<BranchData, InternodeData, BudData>::RefBranch(int handle) {
        assert(handle > 0 && handle < m_branches.size());
        return m_branches[handle];
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    Internode<InternodeData, BudData> &Plant<BranchData, InternodeData, BudData>::RefInternode(int handle) {
        assert(handle > 0 && handle < m_internodes.size());
        return m_internodes[handle];
    }
    template<typename BranchData, typename InternodeData, typename BudData>
    std::vector<BranchHandle> Plant<BranchData, InternodeData, BudData>::GetSortedBranchList() {
        std::vector<BranchHandle> sortedBranches;
        std::queue<BranchHandle> waitList;
        waitList.push(0);
        while (!waitList.empty()) {
            sortedBranches.emplace_back(waitList.front());
            waitList.pop();
            for (const auto &i: m_branches[sortedBranches.back()].m_children) {
                waitList.push(i);
            }
        }
        return sortedBranches;
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    std::vector<InternodeHandle> Plant<BranchData, InternodeData, BudData>::GetSortedInternodeList() {
        std::vector<InternodeHandle> sortedInternodes;
        std::queue<InternodeHandle> waitList;
        waitList.push(0);
        while (!waitList.empty()) {
            sortedInternodes.emplace_back(waitList.front());
            waitList.pop();
            for (const auto &i: m_internodes[sortedInternodes.back()].m_children) {
                waitList.push(i);
            }
        }
        return sortedInternodes;
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    InternodeHandle Plant<BranchData, InternodeData, BudData>::Extend(int targetHandle) {
        assert(targetHandle < m_internodes.size());
        auto &targetInternode = m_internodes[targetHandle];
        assert(!targetInternode.m_recycled);
        assert(targetInternode.m_branchHandle < m_branches.size());
        auto &branch = m_branches[targetInternode.m_branchHandle];
        assert(!branch.m_recycled);

        auto newInternodeHandle = AllocateInternode();
        SetParentInternode(newInternodeHandle, targetHandle);
        auto &newInternode = m_internodes[targetHandle];

        //If this internode is end node, we simply add this to current branch.
        if (targetInternode.m_endNode) {
            targetInternode.m_endNode = false;
            branch.m_internodes.emplace_back(newInternodeHandle);
            newInternode.m_branchHandle = targetInternode.m_branchHandle;
        } else {
            //If this internode is not the end node, we need to create a new branch from the original branch.
            auto newBranchHandle = AllocateBranch();
            auto &newBranch = m_branches[newBranchHandle];
            SetParentBranch(newBranchHandle, targetInternode.m_branchHandle);
            newInternode.m_branchHandle = newBranchHandle;
            newBranch.m_internodes.emplace_back(newInternodeHandle);
        }

        return newInternodeHandle;
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::PruneBranch(int handle) {
        assert(handle != 0);
        assert(!m_branches[handle].m_recycled);
        auto &branch = m_branches[handle];
        //Remove children
        auto children = branch.m_children;
        for (const auto &child: children) {
            PruneBranch(child);
        }
        //Remove internodes
        if (!branch.m_internodes.empty()) {
            //Detach first internode from parent.
            auto &firstInternode = m_internodes[branch.m_internodes.front()];
            if (firstInternode.m_parent != -1)
                DetachChildInternode(firstInternode.m_parent, branch.m_internodes.front());
            auto internodes = branch.m_internodes;
            for (const auto &i: internodes) {
                RecycleInternodeSingle(i);
            }
        }
        //Detach from parent
        if (branch.m_parent != -1) DetachChildBranch(branch.m_parent, handle);
        RecycleBranchSingle(handle);
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::PruneInternode(int handle) {
        assert(handle != 0);
        assert(!m_internodes[handle].m_recycled);
        auto &internode = m_internodes[handle];
        auto &branch = m_branches[internode.m_branchHandle];
        if (handle == branch.m_internodes[0]) {
            PruneBranch(internode.m_branchHandle);
            return;
        }
        //Collect list of subsequent internodes
        std::vector<InternodeHandle> subsequentInternodes;
        for (int i = branch.m_internodes.size() - 1; i > 0; i--) {
            subsequentInternodes.emplace_back(branch.m_internodes[i]);
            if (branch.m_internodes[i] == handle) break;
        }
        assert(subsequentInternodes.size() < branch.m_internodes.size());
        //Detach from parent
        if (internode.m_parent != -1) DetachChildInternode(internode.m_parent, handle);
        //From end node remove one by one.
        for (const auto &i: subsequentInternodes) {
            auto children = m_internodes[i].m_children;
            for (const auto &childInternodeHandle: children) {
                auto branchHandle = m_internodes[childInternodeHandle].m_branchHandle;
                if (branchHandle != internode.m_branchHandle) {
                    PruneBranch(branchHandle);
                }
            }
            RecycleInternodeSingle(i);
        }
    }

#pragma endregion
#pragma region Internal

    template<typename BudData>
    Bud<BudData>::Bud() {
        m_data = {};
    }

    template<typename InternodeData, typename BudData>
    Internode<InternodeData, BudData>::Internode(int handle) {
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

    template<typename BranchData, typename InternodeData, typename BudData>
    Plant<BranchData, InternodeData, BudData>::Plant() {
        AllocateBranch();
        AllocateInternode();
        auto &rootBranch = m_branches[0];
        auto &rootInternode = m_internodes[0];
        rootInternode.m_branchHandle = 0;
        rootBranch.m_internodes.emplace_back(0);
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::DetachChildInternode(int targetHandle, int childHandle) {
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

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::SetParentInternode(int targetHandle, int parentHandle) {
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

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::DetachChildBranch(int targetHandle, int childHandle) {
        assert(targetHandle >= 0 && childHandle >= 0 && targetHandle < m_branches.size() &&
               childHandle < m_branches.size());
        auto &targetBranch = m_branches[targetHandle];
        auto &childBranch = m_branches[childHandle];
        assert(!targetBranch.m_recycled);
        assert(!childBranch.m_recycled);
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

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::SetParentBranch(int targetHandle, int parentHandle) {
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

    template<typename BranchData, typename InternodeData, typename BudData>
    BranchHandle Plant<BranchData, InternodeData, BudData>::AllocateBranch() {
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

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::RecycleBranchSingle(int handle) {
        assert(!m_branches[handle].m_recycled);
        auto &branch = m_branches[handle];
        branch.m_parent = -1;
        branch.m_children.clear();
        branch.m_internodes.clear();

        branch.m_recycled = true;
        m_branchPool.emplace(handle);
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    void Plant<BranchData, InternodeData, BudData>::RecycleInternodeSingle(int handle) {
        assert(!m_internodes[handle].m_recycled);
        auto &internode = m_internodes[handle];
        internode.m_parent = -1;
        internode.m_branchHandle = -1;
        internode.m_endNode = true;
        internode.m_children.clear();
        internode.m_buds.clear();

        internode.m_recycled = true;
        m_internodePool.emplace(handle);
    }

    template<typename BranchData, typename InternodeData, typename BudData>
    InternodeHandle Plant<BranchData, InternodeData, BudData>::AllocateInternode() {
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




#pragma endregion
}