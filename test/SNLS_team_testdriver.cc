#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

using namespace std;

#include "SNLS_config.h"

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)

#include "SNLS_device_forall.h"
#include "SNLS_lup_solve.h"
#include "SNLS_TrDLDenseG.h"
#include "SNLS_testmodels.h"

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

namespace {

/**
 * @brief Builds a small, deterministic, non-singular test problem: an n*n
 * row-major matrix with a spread of off-diagonal values, deliberately
 * arranged so the first pivot step forces at least one row swap (row n-1
 * has the largest-magnitude entry in column 0), plus a right-hand-side
 * vector. Used by every bit-identical-vs-serial test below so they all
 * exercise both the swap-tracking and elimination logic, not just a
 * diagonally-dominant, never-pivots case.
 */
template <int n>
void buildLupTestProblem(std::vector<double>& a, std::vector<double>& b)
{
   a.assign(n * n, 0.0);
   b.assign(n, 0.0);
   for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
         a[i*n + j] = 1.0 + 0.5 * ((i * 7 + j * 3) % 5);
      }
      a[i*n + i] += 10.0 + i;
      b[i] = 1.0 + 2.0 * i - 0.3 * (i * i);
   }
   a[(n - 1) * n + 0] = 200.0; // forces a pivot swap at i==0
}

/**
 * @brief Design doc §5 case 7: the cooperative SNLS_LUP_Decompose/
 * SNLS_LUP_Solve overloads, called with tid=0/nthreads=1, must produce
 * bit-for-bit identical results to the existing, untouched serial
 * overloads on the same input -- validating both the swap/pivot-vector
 * bookkeeping (including p[n]'s swap count) and the decomposed matrix and
 * solution values.
 *
 * The serial overload represents a pivot-row swap by repointing row
 * pointers (`mtx[i] = mtx[imax]`), leaving the flat backing array's
 * physical byte order untouched; the cooperative overload, having no
 * pointer-indirection layer available in shared memory, physically swaps
 * row *contents* in the flat array instead (both are valid
 * implementations of the identical documented pivot-vector semantics).
 * So the decomposed-matrix comparison below reads the serial side through
 * `mtx[i][j]` (its logical, post-swap view) rather than the flat
 * `aSerial` buffer directly, to compare like with like.
 */
template <int n>
void expectBitIdenticalDecomposeSolve()
{
   std::vector<double> aSerial, aCoop, b;
   buildLupTestProblem<n>(aSerial, b);
   aCoop = aSerial;

   double* mtx[n];
   for (int i = 0, k = 0; i < n; ++i, k += n) { mtx[i] = aSerial.data() + k; }
   int pivSerial[n + 1];
   int errSerial = ::SNLS_LUP_Decompose<n>(mtx, pivSerial, 1e-50);
   ASSERT_EQ(errSerial, 0);
   std::vector<double> xSerial(n);
   ::SNLS_LUP_Solve<n>(mtx, pivSerial, xSerial.data(), b.data());

   int pivCoop[n + 1];
   int errCoop = ::SNLS_LUP_Decompose<n>(aCoop.data(), pivCoop, 1e-50,
                                         /*tid=*/0, /*nthreads=*/1);
   ASSERT_EQ(errCoop, 0);
   std::vector<double> xCoop(n);
   ::SNLS_LUP_Solve<n>(aCoop.data(), pivCoop, xCoop.data(), b.data(),
                       /*tid=*/0, /*nthreads=*/1);

   for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
         EXPECT_EQ(mtx[i][j], aCoop[i*n + j]) << "n=" << n << " a[" << i << "][" << j << "]";
      }
   }
   for (int i = 0; i <= n; ++i) {
      EXPECT_EQ(pivSerial[i], pivCoop[i]) << "n=" << n << " p[" << i << "]";
   }
   for (int i = 0; i < n; ++i) {
      EXPECT_EQ(xSerial[i], xCoop[i]) << "n=" << n << " x[" << i << "]";
   }
}

/**
 * @brief Same claim as expectBitIdenticalDecomposeSolve(), but through the
 * composite (decompose-then-solve) entry points instead of the two-array
 * ones -- the shape SNLSTrDlDenseG/SNLSTrDlDenseG_Batch actually call.
 *
 * Only `x` is compared here: the serial composite overload owns its pivot
 * vector as a private local, so (unlike the two-array test above) there
 * is no way for this test to recover the serial side's logical row
 * ordering to compare `a` meaningfully. `x` is the behaviorally-relevant
 * invariant anyway -- it is all any real caller (e.g. computeNewtonStep)
 * ever reads.
 */
template <int n>
void expectBitIdenticalComposite()
{
   std::vector<double> aSerial, aCoop, b;
   buildLupTestProblem<n>(aSerial, b);
   aCoop = aSerial;

   std::vector<double> xSerial(n), xCoop(n);
   int errSerial = ::SNLS_LUP_Solve<n>(aSerial.data(), xSerial.data(), b.data(), 1e-50);
   int pivCoop[n + 1];
   int errCoop = ::SNLS_LUP_Solve<n>(aCoop.data(), pivCoop, xCoop.data(), b.data(),
                                     /*tid=*/0, /*nthreads=*/1, 1e-50);
   ASSERT_EQ(errSerial, 0);
   ASSERT_EQ(errCoop, 0);
   for (int i = 0; i < n; ++i) { EXPECT_EQ(xSerial[i], xCoop[i]) << "n=" << n; }
}

/**
 * @brief Same claim again, through SNLS_LUP_SolveX (multi-right-hand-side)
 * -- design doc §5 case 6/7 combined: several right-hand sides against
 * one matrix, decomposed once, compared bit-for-bit against calling the
 * existing serial SNLS_LUP_SolveX. Only `xb` is compared, for the same
 * reason as expectBitIdenticalComposite() above.
 */
template <int n>
void expectBitIdenticalSolveX()
{
   std::vector<double> aSerial, b;
   buildLupTestProblem<n>(aSerial, b);
   std::vector<double> aCoop = aSerial;

   constexpr int nRHS = 3;
   std::vector<double> xbSerial(nRHS * n), xbCoop(nRHS * n);
   for (int r = 0; r < nRHS; ++r) {
      for (int i = 0; i < n; ++i) {
         double v = b[i] + r * 0.7 - i * 0.2;
         xbSerial[r*n + i] = v;
         xbCoop[r*n + i]   = v;
      }
   }

   int errSerial = ::SNLS_LUP_SolveX<n>(aSerial.data(), xbSerial.data(), nRHS, 1e-50);
   int pivCoop[n + 1];
   int errCoop = ::SNLS_LUP_SolveX<n>(aCoop.data(), pivCoop, xbCoop.data(), nRHS,
                                      /*tid=*/0, /*nthreads=*/1, 1e-50);
   ASSERT_EQ(errSerial, 0);
   ASSERT_EQ(errCoop, 0);
   for (int i = 0; i < nRHS * n; ++i) { EXPECT_EQ(xbSerial[i], xbCoop[i]) << "n=" << n; }
}

} // namespace

TEST(LupSolveCooperative, BitIdenticalToSerialAtNThreadsOne)
{
   expectBitIdenticalDecomposeSolve<3>();
   expectBitIdenticalDecomposeSolve<5>();
   expectBitIdenticalDecomposeSolve<8>();
}

TEST(LupSolveCooperative, CompositeBitIdenticalToSerialAtNThreadsOne)
{
   expectBitIdenticalComposite<3>();
   expectBitIdenticalComposite<5>();
   expectBitIdenticalComposite<8>();
}

TEST(LupSolveCooperative, SolveXBitIdenticalToSerialAtNThreadsOne)
{
   expectBitIdenticalSolveX<3>();
   expectBitIdenticalSolveX<5>();
   expectBitIdenticalSolveX<8>();
}

/**
 * @brief A genuinely singular (not masked-off/dummy) matrix must be
 * reported as a failure by the cooperative SNLS_LUP_Decompose exactly
 * like the existing serial one.
 */
TEST(LupSolveCooperative, SingularMatrixReportsFailureLikeSerial)
{
   constexpr int n = 3;
   std::vector<double> aSerial(n * n, 0.0), aCoop(n * n, 0.0);

   double* mtx[n];
   for (int i = 0, k = 0; i < n; ++i, k += n) { mtx[i] = aSerial.data() + k; }
   int pivSerial[n + 1];
   EXPECT_LT(::SNLS_LUP_Decompose<n>(mtx, pivSerial, 1e-50), 0);

   int pivCoop[n + 1];
   EXPECT_LT(::SNLS_LUP_Decompose<n>(aCoop.data(), pivCoop, 1e-50, /*tid=*/0, /*nthreads=*/1), 0);
}

/**
 * @brief Design doc §3.5: the elimination step's row stride
 * (`for (int j = i+1+tid; j<n; j+=nthreads)`) must partition the
 * remaining rows `{i+1, ..., n-1}` across tid=0..nthreads-1 with no gaps
 * and no overlaps, for NTEAM smaller than, equal to, and larger than the
 * number of remaining rows. Pure host arithmetic -- validates the
 * indexing math directly rather than needing real concurrent execution.
 */
TEST(LupSolveCooperative, EliminationRowStrideCoverage)
{
   struct Combo { int n; int i; int nthreads; };
   const std::vector<Combo> combos = {
      {8, 0, 3},   // doesn't divide evenly
      {8, 0, 8},   // NTEAM == number of remaining rows
      {8, 0, 24},  // NTEAM > remaining rows -- most threads idle
      {8, 5, 4},   // few rows remain, late pivot step
      {5, 4, 3},   // zero rows remain (last pivot step)
      {3, 0, 1},   // serial-equivalent case
   };

   for (const auto& c : combos) {
      std::set<int> covered;
      for (int tid = 0; tid < c.nthreads; ++tid) {
         for (int j = c.i + 1 + tid; j < c.n; j += c.nthreads) {
            ASSERT_TRUE(covered.insert(j).second)
               << "row " << j << " covered more than once (n=" << c.n
               << ", i=" << c.i << ", nthreads=" << c.nthreads << ")";
         }
      }
      const int expected = std::max(0, c.n - (c.i + 1));
      EXPECT_EQ((int)covered.size(), expected);
      for (int j = c.i + 1; j < c.n; ++j) {
         EXPECT_EQ(covered.count(j), 1u) << "row " << j << " never covered";
      }
   }
}

/**
 * @brief Design doc §4.3/§5 case 6: SNLS_LUP_SolveX's right-hand-side
 * distribution stride (`for (int r = tid; r<nRHS; r+=nthreads)`) must
 * partition `{0, ..., nRHS-1}` across tid=0..nthreads-1 with no gaps and
 * no overlaps, for nRHS smaller than, equal to, and larger than nthreads.
 */
TEST(LupSolveCooperative, SolveXRHSStrideCoverage)
{
   struct Combo { int nRHS; int nthreads; };
   const std::vector<Combo> combos = {
      {7, 3}, {7, 8}, {1, 4}, {10, 1}, {12, 4},
   };
   for (const auto& c : combos) {
      std::set<int> covered;
      for (int tid = 0; tid < c.nthreads; ++tid) {
         for (int r = tid; r < c.nRHS; r += c.nthreads) {
            ASSERT_TRUE(covered.insert(r).second)
               << "RHS " << r << " covered more than once (nRHS=" << c.nRHS
               << ", nthreads=" << c.nthreads << ")";
         }
      }
      EXPECT_EQ((int)covered.size(), c.nRHS);
      for (int r = 0; r < c.nRHS; ++r) {
         EXPECT_EQ(covered.count(r), 1u) << "RHS " << r << " never covered";
      }
   }
}

/**
 * @brief Design doc §5 case 6: SNLS_LUP_SolveX with several right-hand
 * sides, for nRHS values both smaller and larger than the simulated
 * thread count.
 *
 * True concurrent execution needs GPU hardware (§5/§7), so this decomposes
 * once via the already-validated cooperative SNLS_LUP_Decompose (tid=0,
 * nthreads=1), then hands each right-hand side's substitution to a
 * simulated "thread" per the same `r=tid; r<nRHS; r+=nthreads` stride
 * SNLS_LUP_SolveX itself uses internally -- via the plain, teamSync()-free
 * serial substitution routine, exactly as SNLS_LUP_SolveX does (see
 * SNLS_lup_solve.h's cooperative SNLS_LUP_SolveX doc comment for why it
 * uses that routine rather than the cooperative one for this). The
 * combined result across all simulated threads must match calling the
 * existing serial SNLS_LUP_SolveX once for every right-hand side.
 */
TEST(LupSolveCooperative, SolveXSimulatedMultiThreadDistribution)
{
   constexpr int n = 5;
   struct Combo { int nRHS; int nthreads; };
   const std::vector<Combo> combos = {
      {2, 4},  // nRHS < nthreads
      {5, 5},  // nRHS == nthreads
      {7, 3},  // nRHS > nthreads, doesn't divide evenly
      {9, 4},  // nRHS > nthreads
   };

   for (const auto& c : combos) {
      std::vector<double> aRef, b;
      buildLupTestProblem<n>(aRef, b);
      std::vector<double> aSim = aRef;

      std::vector<double> xbRef(c.nRHS * n);
      for (int r = 0; r < c.nRHS; ++r) {
         for (int i = 0; i < n; ++i) { xbRef[r*n + i] = b[i] + r * 0.7 - i * 0.2; }
      }
      int errRef = ::SNLS_LUP_SolveX<n>(aRef.data(), xbRef.data(), c.nRHS, 1e-50);
      ASSERT_EQ(errRef, 0);

      int pivSim[n + 1];
      int errSim = ::SNLS_LUP_Decompose<n>(aSim.data(), pivSim, 1e-50, /*tid=*/0, /*nthreads=*/1);
      ASSERT_EQ(errSim, 0);

      double* mtx[n];
      for (int i = 0, k = 0; i < n; ++i, k += n) { mtx[i] = aSim.data() + k; }

      std::vector<double> xbSim(c.nRHS * n);
      for (int r = 0; r < c.nRHS; ++r) {
         for (int i = 0; i < n; ++i) { xbSim[r*n + i] = b[i] + r * 0.7 - i * 0.2; }
      }
      for (int tid = 0; tid < c.nthreads; ++tid) {
         for (int r = tid; r < c.nRHS; r += c.nthreads) {
            double* xThis = &xbSim[r*n];
            double wrk[n];
            for (int k = 0; k < n; ++k) { wrk[k] = xThis[k]; }
            ::SNLS_LUP_Solve<n>(mtx, pivSim, xThis, wrk);
         }
      }

      for (int i = 0; i < c.nRHS * n; ++i) {
         EXPECT_EQ(xbRef[i], xbSim[i])
            << "nRHS=" << c.nRHS << " nthreads=" << c.nthreads << " i=" << i;
      }
   }
}

/**
 * @brief Design doc §5 case 3: heterogeneous iteration count within
 * separate solveTeam() calls -- each point converges after a different
 * number of Newton iterations, driven entirely through the real
 * snls::forall_team + SNLSTrDlDenseG::solveTeam() path (not a synthetic
 * stand-in), exercised via forall_team's CPU fallback.
 *
 * This validates the actual, real consensus-driven loop inside
 * solveImpl() end to end; it cannot exercise genuine cross-team
 * teamSync() divergence (the CPU fallback never has more than one team
 * sharing a call), which needs GPU hardware -- see §5/§7.
 */
TEST(SolveTeamNonBatch, HeterogeneousIterationCountThroughForallTeam)
{
   constexpr double lambda = 0.9999;
   constexpr int nDimSys = 4;
   const int npts = 15;

   // A deliberate difficulty gradient rather than narrow random noise:
   // points near the "easy" starting guess (x=0, per SNLS_testmodels.h's
   // own comment on this problem) converge in fewer Newton iterations
   // than ones near the "standard," harder starting guess (x=-1) or
   // beyond -- confirmed empirically to actually vary (asserted below),
   // not just assumed.
   std::vector<double> x0(npts * nDimSys);
   for (int i = 0; i < npts; ++i) {
      const double scale = -1.5 * (static_cast<double>(i) / (npts - 1));
      for (int k = 0; k < nDimSys; ++k) { x0[i*nDimSys + k] = scale; }
   }

   std::vector<int> itersUsed(npts, -1);
   std::vector<int> converged(npts, 0); // not std::vector<bool> -- no .data()
   const double* x0ptr = x0.data();
   int* itersPtr = itersUsed.data();
   int* convPtr = converged.data();

   snls::forall_team<64, 4>(0, npts,
      [=] __snls_hdev__ (int i, int tid, int nthreads, int teamBase, auto& consensus) {
         auto crj = broyden_lambda(lambda, nDimSys);
         snls::SNLSTrDlDenseG<decltype(crj), nDimSys> solver(crj);
         snls::TrDeltaControl deltaControl;
         deltaControl._deltaInit = 1.0;
         solver.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 0);
         for (int k = 0; k < nDimSys; ++k) { solver._x[k] = x0ptr[i*nDimSys + k]; }
         snls::SNLSStatus_t status = solver.solveTeam(tid, nthreads, teamBase, consensus);
         convPtr[i]  = snls::isConverged(status);
         itersPtr[i] = solver.getNFEvals();
      });

   int minIters = itersUsed[0];
   int maxIters = itersUsed[0];
   for (int i = 0; i < npts; ++i) {
      EXPECT_TRUE(converged[i]) << "point " << i << " failed to converge";
      minIters = std::min(minIters, itersUsed[i]);
      maxIters = std::max(maxIters, itersUsed[i]);
   }
   // Confirm genuine heterogeneity in iteration count across points --
   // otherwise this would not actually be exercising different-iteration-
   // count teams at all.
   EXPECT_GT(maxIters, minIters);
}

/**
 * @brief Design doc §5 cases 4 & 5: a sequence of two different-sized
 * solves for "the same point" (mirroring ExaCMech's 24x24-then-3x3-...
 * scenario in miniature), run both with solveTeam()'s self-managed
 * scratch (extScratch/extPiv == nullptr) and with one caller-owned buffer
 * sized to the larger problem and reused across both solves in the
 * sequence. Both must converge to the same, correct answer, and the
 * two scratch-management strategies must produce bit-identical results.
 */
TEST(SolveTeamNonBatch, DifferentSizedSequenceSelfManagedVsSharedScratch)
{
   constexpr double lambda = 0.9999;
   constexpr int nDimA = 3;
   constexpr int nDimB = 6;

   auto crjA = broyden_lambda(lambda, nDimA);
   auto crjB = broyden_lambda(lambda, nDimB);

   auto runOnce = [&](double* extScratchA, int* extPivA,
                       double* extScratchB, int* extPivB,
                       double* xAOut, double* xBOut,
                       snls::SNLSStatus_t& statusAOut, snls::SNLSStatus_t& statusBOut) {
      snls::TrDeltaControl deltaControl;
      deltaControl._deltaInit = 1.0;

      snls::SNLSTrDlDenseG<decltype(crjA), nDimA> solverA(crjA);
      solverA.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 0);
      for (int k = 0; k < nDimA; ++k) { solverA._x[k] = 0.0; }
      snls::TeamActivityConsensus<1> consensusA;
      statusAOut = solverA.solveTeam(0, 1, 0, consensusA, extScratchA, extPivA);
      solverA.getX(xAOut);

      snls::SNLSTrDlDenseG<decltype(crjB), nDimB> solverB(crjB);
      solverB.setupSolver(NL_MAXITER, NL_TOLER, &deltaControl, 0);
      for (int k = 0; k < nDimB; ++k) { solverB._x[k] = 0.0; }
      snls::TeamActivityConsensus<1> consensusB;
      statusBOut = solverB.solveTeam(0, 1, 0, consensusB, extScratchB, extPivB);
      solverB.getX(xBOut);
   };

   double xASelf[nDimA], xBSelf[nDimB];
   snls::SNLSStatus_t statusASelf, statusBSelf;
   runOnce(nullptr, nullptr, nullptr, nullptr, xASelf, xBSelf, statusASelf, statusBSelf);

   // One buffer sized to the LARGER problem, reused for BOTH solves in
   // sequence (design doc §3.5).
   constexpr int maxNXnDim = nDimB * nDimB;
   constexpr int maxNDim   = nDimB;
   double sharedScratch[maxNXnDim + maxNDim];
   int    sharedPiv[maxNDim + 1];

   double xAShared[nDimA], xBShared[nDimB];
   snls::SNLSStatus_t statusAShared, statusBShared;
   runOnce(sharedScratch, sharedPiv, sharedScratch, sharedPiv,
           xAShared, xBShared, statusAShared, statusBShared);

   EXPECT_TRUE(snls::isConverged(statusASelf));
   EXPECT_TRUE(snls::isConverged(statusBSelf));
   EXPECT_EQ(statusASelf, statusAShared);
   EXPECT_EQ(statusBSelf, statusBShared);
   for (int k = 0; k < nDimA; ++k) { EXPECT_EQ(xASelf[k], xAShared[k]) << "A k=" << k; }
   for (int k = 0; k < nDimB; ++k) { EXPECT_EQ(xBSelf[k], xBShared[k]) << "B k=" << k; }
}

#endif // SNLS_RAJA_PORT_SUITE
