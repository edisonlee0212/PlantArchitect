#pragma once

#include "plant_architect_export.h"

#define InternodeHandle int
#define BranchHandle int
using namespace UniEngine;
namespace Orchards {
    class Internode {
        friend class Plant;
        bool m_endNode = true;
        bool m_recycled = false;
        InternodeHandle m_handle = 0;
        BranchHandle m_branchHandle = 0;

        InternodeHandle m_parent = 0;
        std::vector<InternodeHandle> m_children;

        GlobalTransform m_globalTransform;
    public:
        explicit Internode(InternodeHandle handle);

        Transform m_transform;
    };

    class Branch {
        friend class Plant;

        bool m_recycled = false;
        BranchHandle m_handle = 0;
        std::vector<InternodeHandle> m_internodes;

        BranchHandle m_parent = 0;
        std::vector<BranchHandle> m_children;

        GlobalTransform m_globalTransform;
    public:
        explicit Branch(BranchHandle handle);

        Transform m_transform;
    };

    class Plant {
        std::vector<Branch> m_branches;
        std::vector<Internode> m_internodes;
        std::queue<InternodeHandle> m_internodePool;
        std::queue<BranchHandle> m_branchPool;

        InternodeHandle AllocateInternode();

        void RecycleInternode(InternodeHandle handle);

        BranchHandle AllocateBranch();

        void RecycleBranch(BranchHandle handle);

    public:
        void SetParentBranch(BranchHandle targetHandle, BranchHandle parentHandle);

        void RemoveChildBranch(BranchHandle targetHandle, BranchHandle childHandle);

        void SetParentInternode(InternodeHandle targetHandle, InternodeHandle parentHandle);

        void RemoveChildInternode(InternodeHandle targetHandle, InternodeHandle childHandle);

        InternodeHandle Extend(InternodeHandle targetHandle);
    };
}