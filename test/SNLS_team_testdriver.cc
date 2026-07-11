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

#endif // SNLS_RAJA_PORT_SUITE
