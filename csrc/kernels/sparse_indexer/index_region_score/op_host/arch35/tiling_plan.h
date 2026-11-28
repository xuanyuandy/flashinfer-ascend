// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef INDEXER_REGION_SCORE_TILING_PLAN_H
#define INDEXER_REGION_SCORE_TILING_PLAN_H

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstddef>
#include <limits>
#include "../../include/arch35/tiling_data_def.h"

namespace IndexerRegionScoreHost {
// This cap applies to the IRS algorithm layout, including its platform library
// prefix, bounded metadata, Score pool and allocation alignment reserve. Outer
// Contiguous/ViewCopy allocations in the ACLNN executor are accounted separately.
constexpr uint64_t kTopKOnlyWorkspaceBudgetBytes = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t kBoundedExecutorReserve = 4096U;
struct BoundedWorkspacePlan {
    uint32_t batchTokens = 0U;
    uint64_t regionStride = 0U;
    uint64_t scratchOffset = 0U;
    uint64_t scorePoolOffset = 0U;
    uint64_t totalBytes = 0U;
};

inline bool PlanBoundedWorkspace(uint64_t budget, uint64_t libraryBytes,
    uint64_t fixedAuxBytes, uint32_t groups, uint32_t tokens, uint32_t regions,
    uint32_t topK, uint32_t qMicroTile, BoundedWorkspacePlan &plan)
{
    plan = {};
    if (groups == 0U || tokens == 0U || regions > 16777216U || topK == 0U || topK > 4096U ||
        libraryBytes > budget || fixedAuxBytes > budget - libraryBytes) {
        return false;
    }
    uint64_t prefix = libraryBytes + fixedAuxBytes;
    if (prefix > budget || budget - prefix < kBoundedExecutorReserve + 31U) {
        return false;
    }
    prefix = (prefix + 31U) & ~uint64_t(31U);
    const uint64_t available = budget - kBoundedExecutorReserve - prefix;
    // R=0 still needs a positive physical stride; there are no valid Score reads.
    plan.regionStride = std::max(uint64_t(8U), (uint64_t(regions) + 7U) & ~uint64_t(7U));
    const uint64_t scratchRowBytes = topK % 8U == 0U ? 0U :
        ((uint64_t(topK) + 7U) & ~uint64_t(7U)) * sizeof(int32_t);
    const uint64_t bytesPerToken = uint64_t(groups) *
        (plan.regionStride * sizeof(float) + scratchRowBytes);
    if (bytesPerToken == 0U || available / bytesPerToken == 0U) {
        return false;
    }
    uint64_t batchTokens = std::min(uint64_t(tokens), available / bytesPerToken);
    // Keep the entire input as one batch if it fits; otherwise preserve the
    // Q micro geometry when the budget admits at least one complete micro.
    if (batchTokens < tokens && qMicroTile > 1U && batchTokens >= qMicroTile) {
        batchTokens = batchTokens / qMicroTile * qMicroTile;
    }
    plan.batchTokens = static_cast<uint32_t>(batchTokens);
    plan.scratchOffset = prefix - libraryBytes;
    const uint64_t scratchBytes = uint64_t(groups) * batchTokens * scratchRowBytes;
    plan.scorePoolOffset = plan.scratchOffset + scratchBytes;
    plan.totalBytes = prefix + scratchBytes +
        uint64_t(groups) * batchTokens * plan.regionStride * sizeof(float);
    return plan.totalBytes <= budget - kBoundedExecutorReserve &&
        plan.totalBytes <= std::numeric_limits<size_t>::max();
}

// The weighted runner reserves its plan before the existing data workspace.
// Reduce only the available budget so both runners retain the same data offsets.
inline bool PlanBoundedScoreWorkspace(uint64_t budget, uint64_t libraryBytes,
    uint64_t fixedAuxBytes, uint32_t groups, uint32_t tokens, uint32_t regions,
    uint32_t topK, uint32_t qMicroTile, uint32_t requests, uint32_t pairs,
    BoundedWorkspacePlan &plan)
{
    uint64_t reserved = 0U;
    for (uint64_t attempt = 0U; attempt <= static_cast<uint64_t>(pairs); ++attempt) {
        if (reserved > budget || !PlanBoundedWorkspace(budget - reserved, libraryBytes,
                fixedAuxBytes, groups, tokens, regions, topK, qMicroTile, plan)) {
            plan = {};
            return false;
        }
        const uint64_t planBytes = IndexerRegionScore::ScorePlanWorkspaceBytes(
            requests, pairs, tokens, plan.batchTokens);
        if (planBytes <= reserved) {
            plan.totalBytes += planBytes;
            return true;
        }
        // A smaller batch can increase only the bounded number of cut slots.
        reserved = planBytes;
    }
    plan = {};
    return false;
}

constexpr uint32_t kProxyDim = 256;
constexpr uint32_t kCubeMRows = 128;
constexpr uint32_t kKeyTile = 512;

inline uint32_t SelectPagedGroupTile(uint32_t groups, uint32_t heads)
{
    // The second Q micro's L1->L0 offset is addressed in 16-row units.
    // Prefer a tile that keeps every group block (including the last) aligned.
    const uint32_t capacity = std::min(groups, kCubeMRows / heads);
    for (uint32_t count = capacity; count > 0U; --count) {
        if ((count * heads) % 16U == 0U && groups % count == 0U) {
            return count;
        }
    }
    return capacity;
}

inline uint32_t SelectPagedQMicroTile(uint32_t groups, uint32_t heads)
{
    const uint32_t groupTile = SelectPagedGroupTile(groups, heads);
    return std::max(1U, kCubeMRows / (groupTile * heads));
}

struct QkShardPlan {
    uint32_t qShards;
    uint32_t kShards;
};

inline bool PreferTrafficPlan(uint64_t maxTiles, uint64_t traffic, uint32_t cores,
    uint64_t trafficPlanMaxTiles, uint64_t trafficPlanBytes, uint32_t trafficPlanCores)
{
    return maxTiles < trafficPlanMaxTiles ||
                (maxTiles == trafficPlanMaxTiles && traffic < trafficPlanBytes) ||
                (maxTiles == trafficPlanMaxTiles && traffic == trafficPlanBytes &&
                 cores > trafficPlanCores);
}

inline bool PreferGatherPlan(uint64_t maxTiles, uint64_t gatherCommands, uint64_t traffic, uint32_t cores,
    uint64_t gatherPlanMaxTiles, uint64_t gatherPlanCommands, uint64_t gatherPlanBytes, uint32_t gatherPlanCores)
{
    return maxTiles < gatherPlanMaxTiles ||
                (maxTiles == gatherPlanMaxTiles && gatherCommands < gatherPlanCommands) ||
                (maxTiles == gatherPlanMaxTiles && gatherCommands == gatherPlanCommands &&
                 traffic < gatherPlanBytes) ||
                (maxTiles == gatherPlanMaxTiles && gatherCommands == gatherPlanCommands &&
                 traffic == gatherPlanBytes && cores > gatherPlanCores);
}

inline QkShardPlan SelectQkShardPlan(uint32_t qRanges, uint32_t kRanges,
                              uint32_t pairCapacity, uint32_t groups,
                              uint32_t summariesPerPage, uint32_t qMicroTile,
                              uint32_t heads, bool pagedGather, uint32_t keyTile = kKeyTile)
{
    QkShardPlan trafficPlan{1U, 1U};
    QkShardPlan gatherPlan{1U, 1U};
    uint64_t trafficPlanMaxTiles = std::numeric_limits<uint64_t>::max();
    uint64_t trafficPlanBytes = std::numeric_limits<uint64_t>::max();
    uint64_t trafficPlanGatherCommands = std::numeric_limits<uint64_t>::max();
    uint32_t trafficPlanCores = 0U;
    uint64_t gatherPlanMaxTiles = std::numeric_limits<uint64_t>::max();
    uint64_t gatherPlanBytes = std::numeric_limits<uint64_t>::max();
    uint64_t gatherPlanCommands = std::numeric_limits<uint64_t>::max();
    uint32_t gatherPlanCores = 0U;
    const uint32_t maxQShards = std::min(qRanges, pairCapacity);
    for (uint32_t qShards = 1U; qShards <= maxQShards; ++qShards) {
        const uint32_t maxKShards = std::min(kRanges, pairCapacity / qShards);
        for (uint32_t kShards = 1U; kShards <= maxKShards; ++kShards) {
            const uint64_t maxTiles =
                static_cast<uint64_t>((qRanges + qShards - 1U) / qShards) *
                ((kRanges + kShards - 1U) / kShards);
            // Both operands are FP8.  Q is re-read by every K shard, while
            // paged K is re-gathered by every Q shard.  Constants common to
            // every candidate are omitted without changing the ordering.
            const uint64_t qTraffic = static_cast<uint64_t>(kShards) * qRanges *
                groups * qMicroTile * heads * kProxyDim;
            const uint64_t kTraffic = static_cast<uint64_t>(qShards) * kRanges *
                keyTile * kProxyDim;
            const uint64_t traffic = qTraffic + kTraffic;
            const uint64_t gatherCommands = static_cast<uint64_t>(qShards) * kRanges *
                ((keyTile + summariesPerPage - 1U) / summariesPerPage);
            const uint32_t cores = qShards * kShards;
            if (PreferTrafficPlan(maxTiles, traffic, cores, trafficPlanMaxTiles, trafficPlanBytes, trafficPlanCores)) {
                trafficPlan = {qShards, kShards};
                trafficPlanMaxTiles = maxTiles;
                trafficPlanBytes = traffic;
                trafficPlanGatherCommands = gatherCommands;
                trafficPlanCores = cores;
            }
            if (PreferGatherPlan(maxTiles, gatherCommands, traffic, cores, gatherPlanMaxTiles, gatherPlanCommands, gatherPlanBytes, gatherPlanCores)) {
                gatherPlan = {qShards, kShards};
                gatherPlanMaxTiles = maxTiles;
                gatherPlanBytes = traffic;
                gatherPlanCommands = gatherCommands;
                gatherPlanCores = cores;
            }
        }
    }
    // Continuous cache has no page-table command-amortization trade-off.  Its
    // best plan is therefore the load-balanced candidate with the least
    // duplicated Q/K traffic.  Paged cache additionally permits the gather
    // command model to win when it retains useful core occupancy.
    if (!pagedGather) {
        return trafficPlan;
    }
    const uint32_t minimumUsefulCores = (pairCapacity * 3U + 3U) / 4U;
    const bool controlsGatherBottleneck = gatherPlanCommands < trafficPlanGatherCommands &&
        gatherPlan.qShards >= 2U && gatherPlanCores >= minimumUsefulCores &&
        gatherPlanBytes <= trafficPlanBytes * 2U;
    return controlsGatherBottleneck ? gatherPlan : trafficPlan;
}

struct PagedShardPlan {
    uint32_t qMicroTile;
    QkShardPlan shards;
};

struct RequestQueryWork {
    uint64_t microTiles = 0U;
    uint64_t requestSegments = 0U;
};

inline uint64_t ShardCeilDiv(uint64_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0U);
}

// Fill idle pair capacity only when the original packed Q-range bound is
// unchanged. Dtype policy belongs to the paged Host caller, not this helper.
inline QkShardPlan FillPagedQueryShards(QkShardPlan original, uint32_t qRanges,
                                       uint32_t kRanges, uint32_t pairCapacity)
{
    if (qRanges == 0U || kRanges == 0U || pairCapacity == 0U ||
        original.qShards == 0U || original.kShards == 0U ||
        original.qShards > qRanges || original.kShards > kRanges ||
        original.qShards > pairCapacity / original.kShards) {
        return original;
    }
    const uint32_t availableQShards = std::min(qRanges, pairCapacity / original.kShards);
    if (availableQShards > original.qShards &&
        ShardCeilDiv(qRanges, availableQShards) == ShardCeilDiv(qRanges, original.qShards)) {
        return {availableQShards, original.kShards};
    }
    return original;
}

// Intersect a packed shard with a run of equal-sized estimated requests.
// Count full requests arithmetically so Host work never scales with N or T.
inline void AddUniformRequestWork(uint64_t begin, uint64_t end, uint32_t requestLength,
                                  uint32_t qMicroTile, RequestQueryWork &work)
{
    if (begin >= end || requestLength == 0U) {
        return;
    }
    const uint64_t first = begin / requestLength;
    const uint64_t last = (end - 1U) / requestLength;
    work.requestSegments += last - first + 1U;
    if (first == last) {
        work.microTiles += ShardCeilDiv(end - begin, qMicroTile);
        return;
    }
    work.microTiles += ShardCeilDiv(requestLength - begin % requestLength, qMicroTile);
    work.microTiles += ShardCeilDiv((end - 1U) % requestLength + 1U, qMicroTile);
    work.microTiles += (last - first - 1U) * ShardCeilDiv(requestLength, qMicroTile);
}

inline RequestQueryWork EstimateRequestQueryWork(uint32_t totalTokens, uint32_t requestCount,
                                                 uint32_t qMicroTile, uint64_t begin, uint64_t end)
{
    RequestQueryWork work;
    if (totalTokens == 0U || requestCount == 0U || qMicroTile == 0U) {
        return work;
    }
    // cu_seq_lens remains device-resident. Estimate balanced nonempty requests;
    // when T < N there are T singleton requests and N-T empty requests.
    const uint32_t nonempty = std::min(totalTokens, requestCount);
    const uint32_t shortLength = totalTokens / nonempty;
    const uint64_t longEnd = static_cast<uint64_t>(totalTokens % nonempty) * (shortLength + 1U);
    AddUniformRequestWork(begin, std::min(end, longEnd), shortLength + 1U, qMicroTile, work);
    if (end > longEnd) {
        AddUniformRequestWork(std::max(begin, longEnd) - longEnd, end - longEnd,
                              shortLength, qMicroTile, work);
    }
    return work;
}

inline PagedShardPlan SelectPagedRequestShardPlan(uint32_t totalTokens, uint32_t requestCount,
                                                  uint32_t kRanges, uint32_t pairCapacity,
                                                  uint32_t groups, uint32_t summariesPerPage,
                                                  uint32_t heads, uint32_t keyTile = kKeyTile,
                                                  bool balancedRequestGrid = false)
{
    const uint32_t baseQTile = SelectPagedQMicroTile(groups, heads);
    const uint32_t baseQRanges = static_cast<uint32_t>(ShardCeilDiv(totalTokens, baseQTile));
    const auto basePlan = SelectQkShardPlan(baseQRanges, std::max(1U, kRanges), pairCapacity,
                                           groups, summariesPerPage, baseQTile, heads, true, keyTile);
    const uint32_t nonempty = std::min(totalTokens, requestCount);
    // Retain the existing large-query strategy and zero-K launch semantics.
    if (nonempty <= 1U || kRanges == 0U || totalTokens / nonempty >= baseQTile) {
        return {baseQTile, basePlan};
    }
    const uint32_t qTile = std::max(1U, totalTokens / nonempty);
    const uint32_t qRanges = static_cast<uint32_t>(ShardCeilDiv(totalTokens, qTile));
    const uint32_t groupBlocks = static_cast<uint32_t>(ShardCeilDiv(groups, SelectPagedGroupTile(groups, heads)));
    QkShardPlan best{1U, 1U};
    uint64_t bestCost = std::numeric_limits<uint64_t>::max();
    uint64_t bestTraffic = std::numeric_limits<uint64_t>::max();
    uint32_t bestCores = 0U;
    uint64_t bestRatioNumerator = 0U;
    uint32_t bestRatioDenominator = 0U;
    using WideProduct = unsigned __int128;
    static_assert(sizeof(WideProduct) * CHAR_BIT >=
                  std::numeric_limits<uint64_t>::digits + std::numeric_limits<uint32_t>::digits,
                  "Request-grid products must fit without overflow");
    WideProduct bestRequestTraffic = 0U;
    for (uint32_t qShards = 1U; qShards <= std::min(qRanges, pairCapacity); ++qShards) {
        uint64_t maxMicroTiles = 0U;
        uint64_t maxRequestSegments = 0U;
        uint64_t totalRequestSegments = 0U;
        for (uint32_t shard = 0U; shard < qShards; ++shard) {
            const uint64_t firstRange = static_cast<uint64_t>(shard) * (qRanges / qShards) +
                std::min(shard, qRanges % qShards);
            const uint64_t rangeCount = qRanges / qShards + (shard < qRanges % qShards);
            const auto work = EstimateRequestQueryWork(totalTokens, requestCount, qTile,
                firstRange * qTile, std::min<uint64_t>(totalTokens, (firstRange + rangeCount) * qTile));
            maxMicroTiles = std::max(maxMicroTiles, work.microTiles);
            maxRequestSegments = std::max(maxRequestSegments, work.requestSegments);
            totalRequestSegments += work.requestSegments;
        }
        for (uint32_t kShards = 1U; kShards <= std::min(kRanges, pairCapacity / qShards); ++kShards) {
            if (balancedRequestGrid) {
                // Both operands are at most uint64 x uint32; the checked type
                // width above guarantees exact products, including at limits.
                const WideProduct lhs = static_cast<WideProduct>(maxMicroTiles) * bestRatioDenominator;
                const WideProduct rhs = static_cast<WideProduct>(bestRatioNumerator) * kShards;
                const WideProduct requestTraffic = static_cast<WideProduct>(totalRequestSegments) * kShards;
                const uint32_t cores = qShards * kShards;
                if (bestRatioDenominator == 0U || lhs < rhs ||
                    (lhs == rhs && (requestTraffic < bestRequestTraffic ||
                     (requestTraffic == bestRequestTraffic && cores > bestCores)))) {
                    best = {qShards, kShards};
                    bestRatioNumerator = maxMicroTiles;
                    bestRatioDenominator = kShards;
                    bestRequestTraffic = requestTraffic;
                    bestCores = cores;
                }
                continue;
            }
            // The first K load of each request has no predecessor to overlap.
            // Charge one macro equivalent per request segment, in addition to
            // micro-tile work. Subsequent K macros can use the existing prefetch.
            const uint64_t cost = maxMicroTiles * groupBlocks * ShardCeilDiv(kRanges, kShards) +
                maxRequestSegments;
            // K is duplicated only when a request crosses a Q shard boundary,
            // not for disjoint requests assigned to different Q shards.
            const uint64_t traffic = totalRequestSegments * kRanges * keyTile * kProxyDim +
                static_cast<uint64_t>(kShards) * totalTokens * groups * heads * kProxyDim;
            const uint32_t cores = qShards * kShards;
            if (cost < bestCost || (cost == bestCost && traffic < bestTraffic) ||
                (cost == bestCost && traffic == bestTraffic && cores > bestCores)) {
                best = {qShards, kShards};
                bestCost = cost;
                bestTraffic = traffic;
                bestCores = cores;
            }
        }
    }
    return {qTile, best};
}

} // namespace IndexerRegionScoreHost
#endif
