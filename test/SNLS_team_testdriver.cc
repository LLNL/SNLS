#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include <gtest/gtest.h>

using namespace std;

#include "SNLS_config.h"

#if defined(SNLS_RAJA_PORT_SUITE)

#include "SNLS_device_forall.h"

/**
 * @brief Case 1 (design doc §5): verify snls::ForallTeamPacking's
 * itemsPerBlock/effectiveBlock derivation against the worked examples in
 * §4.0, including both the evenly-dividing and non-evenly-dividing cases
 * and the NTEAM>NUMBLOCKS edge case.
 *
 * This is pure host arithmetic -- no GPU, no snls::forall_team call, no
 * SNLS solver involved.
 */
TEST(ForallTeam, PackingWorkedExamples)
{
   // Worked example 1 (§4.0): divides evenly, effectiveBlock==NUMBLOCKS.
   {
      using P = snls::ForallTeamPacking<256, 8>;
      EXPECT_EQ(P::itemsPerBlock, 32);
      EXPECT_EQ(P::effectiveBlock, 256);
   }
   // Worked example 2 (§4.0): does not divide evenly, small rounding loss.
   {
      using P = snls::ForallTeamPacking<256, 7>;
      EXPECT_EQ(P::itemsPerBlock, 36);
      EXPECT_EQ(P::effectiveBlock, 252);
   }
   // NTEAM > NUMBLOCKS (§3.2): itemsPerBlock floors to 1, effectiveBlock is
   // forced up to NTEAM, exceeding the original NUMBLOCKS request.
   {
      using P = snls::ForallTeamPacking<8, 24>;
      EXPECT_EQ(P::itemsPerBlock, 1);
      EXPECT_EQ(P::effectiveBlock, 24);
   }
   // NTEAM == NUMBLOCKS -- one team fills the whole block.
   {
      using P = snls::ForallTeamPacking<32, 32>;
      EXPECT_EQ(P::itemsPerBlock, 1);
      EXPECT_EQ(P::effectiveBlock, 32);
   }
}

/**
 * @brief Case 1 (design doc §5, §3.7): split-dispatch coverage. Pure host
 * arithmetic, no GPU needed, no SNLS solver involved at all -- just
 * testing forall_team's internal main+remainder packing math directly,
 * mirroring exactly the fullCount/remainder decomposition forall_team
 * itself performs on the GPU path.
 *
 * Includes several combinations where totalItems is deliberately NOT a
 * multiple of itemsPerBlock -- the common, previously-buggy ragged-tail
 * case (§3.7) that the main+remainder split dispatch exists to avoid.
 */
TEST(ForallTeam, SplitDispatchCoverage)
{
   struct Combo { int totalItems; int itemsPerBlock; int NTEAM; };
   const std::vector<Combo> combos = {
      {256, 32, 8},    // exact multiple -- no remainder
      {250, 32, 8},    // ragged: 250 = 32*7 + 26
      {1,   32, 8},    // fewer items than one full block's worth
      {33,  32, 8},    // one item spills into the remainder
      {1000, 36, 7},   // worked example 2's itemsPerBlock, ragged total
      {5,   1,  24},   // itemsPerBlock==1 (NTEAM>NUMBLOCKS case)
      {0,   32, 8},    // degenerate: no items at all
   };

   for (const auto& c : combos) {
      const int totalItems   = c.totalItems;
      const int itemsPerBlock = c.itemsPerBlock;
      const int NTEAM         = c.NTEAM;

      const int fullCount = itemsPerBlock * (totalItems / itemsPerBlock);
      const int remainder = totalItems - fullCount;

      ASSERT_EQ(fullCount % itemsPerBlock, 0);
      ASSERT_LT(remainder, itemsPerBlock);
      ASSERT_EQ(fullCount + remainder, totalItems);

      std::set<int> covered;
      // Main dispatch: gidx in [0, fullCount*NTEAM).
      for (int gidx = 0; gidx < fullCount * NTEAM; ++gidx) {
         int i = gidx / NTEAM;
         ASSERT_GE(i, 0);
         ASSERT_LT(i, totalItems);
         covered.insert(i);
      }
      // Remainder dispatch: gidx in [0, remainder*NTEAM).
      for (int gidx = 0; gidx < remainder * NTEAM; ++gidx) {
         int i = fullCount + gidx / NTEAM;
         ASSERT_GE(i, 0);
         ASSERT_LT(i, totalItems);
         covered.insert(i);
      }
      ASSERT_EQ((int)covered.size(), totalItems);
   }
}

/**
 * @brief Step 2 (design doc §3.6/§4.2): verify snls::TeamActivityConsensus's
 * NTEAMS==1 specialization degenerates to exactly `myActive`, at every
 * call, with no hidden carried-over state -- the property §4.5's unified
 * solve()/solveTeam() loop depends on for its `nthreads==1` case to behave
 * identically to today's plain solve() loop.
 *
 * Fully testable on this CPU-only machine: NTEAMS==1 is exactly the case
 * the CPU/OpenMP fallback and the remainder dispatch use, and
 * RAJA::LaunchContext::teamSync() is a no-op there, so report()'s real
 * behavior (not just a stand-in) is being exercised.
 */
TEST(TeamActivityConsensus, TrivialDegenerateCase)
{
   snls::TeamActivityConsensus<1> consensus;

   consensus.report(/*tid=*/0, /*teamBase=*/0, /*myActive=*/true);
   EXPECT_TRUE(consensus.anyActive());

   consensus.report(0, 0, true);
   EXPECT_TRUE(consensus.anyActive());

   // Mimic a converging loop: active for a few iterations, then not --
   // anyActive() must track the LATEST report(), not any prior one.
   consensus.report(0, 0, false);
   EXPECT_FALSE(consensus.anyActive());

   consensus.report(0, 0, true);
   EXPECT_TRUE(consensus.anyActive());
}

/**
 * @brief Step 2 (design doc §3.6/§4.2): verify the block-wide OR-reduction
 * formula itself for NTEAMS>1.
 *
 * True concurrent, multi-team execution requires real GPU hardware (per
 * design doc §5/§7 -- forall_team's CPU/OpenMP fallback never actually
 * has more than one team sharing a call, so it cannot exercise this). This
 * test instead simulates the state a block would be in immediately after
 * every team's teamSync()-gated write phase has completed, by directly
 * pre-populating every team-slot but the one under report()'s own tid==0
 * write path, then calling report() for team 0 (whose teamSync() calls
 * are no-ops here on host) to perform the reduction. This validates the
 * OR-reduction arithmetic and the teamBase==0-only gating; it does not
 * validate real concurrent-write safety, which needs §7's GPU hardware
 * pass.
 */
TEST(TeamActivityConsensus, MultiTeamReductionFormula)
{
   constexpr int NTEAMS = 4;

   // All teams but team 0 report inactive -- team 0 alone keeps the block
   // going.
   {
      snls::TeamActivityConsensus<NTEAMS> consensus;
      for (int t = 1; t < NTEAMS; ++t) { consensus.m_active[t] = false; }
      consensus.report(/*tid=*/0, /*teamBase=*/0, /*myActive=*/true);
      EXPECT_TRUE(consensus.anyActive());
   }
   // Every team, including team 0, reports inactive -- block is done.
   {
      snls::TeamActivityConsensus<NTEAMS> consensus;
      for (int t = 1; t < NTEAMS; ++t) { consensus.m_active[t] = false; }
      consensus.report(0, 0, false);
      EXPECT_FALSE(consensus.anyActive());
   }
   // A team other than 0 (the one doing the reduction) is the sole
   // straggler still active.
   {
      snls::TeamActivityConsensus<NTEAMS> consensus;
      consensus.m_active[1] = false;
      consensus.m_active[2] = true;
      consensus.m_active[3] = false;
      consensus.report(0, 0, false);
      EXPECT_TRUE(consensus.anyActive());
   }
}

/**
 * @brief Step 2, integration-level (design doc §4.1/§4.2): exercise
 * snls::forall_team's CPU/OpenMP fallback dispatch end-to-end -- not just
 * TeamActivityConsensus in isolation -- with a body that has its own
 * multi-iteration loop driven entirely through `consensus.report()`/
 * `consensus.anyActive()`, mirroring the shape §4.5 uses for
 * SNLSTrDlDenseG::solveImpl(). Each point converges after a different
 * number of iterations, verifying the consensus-driven loop structure
 * (report every iteration, mask real work by `myActive`, stop only when
 * the whole "block" -- trivially one team on CPU -- is done) produces the
 * correct per-point iteration count.
 */
TEST(ForallTeam, ConsensusDrivenLoopThroughCPUFallback)
{
   const int npts = 25;
   std::vector<int> targetIters(npts);
   std::vector<int> itersDone(npts, 0);
   for (int i = 0; i < npts; ++i) {
      targetIters[i] = 1 + (i % 7); // varies 1..7 across points
   }
   int* target = targetIters.data();
   int* done   = itersDone.data();

   snls::forall_team<64, 4>(0, npts,
      [=] __snls_hdev__ (int i, int tid, int UNUSED(nthreads), int teamBase, auto& consensus) {
         bool myActive = true;
         while (true) {
            consensus.report(tid, teamBase, myActive);
            if (!consensus.anyActive()) { break; }
            if (myActive) {
               done[i] += 1;
               if (done[i] >= target[i]) { myActive = false; }
            }
         }
      });

   for (int i = 0; i < npts; ++i) {
      EXPECT_EQ(itersDone[i], targetIters[i]) << "mismatch at point " << i;
   }
}

#endif // SNLS_RAJA_PORT_SUITE
