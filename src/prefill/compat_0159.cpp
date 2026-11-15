// src/prefill/compat_0159.cpp - the 0.1.59 Prefill entry points that the fork's prefill.cpp does not
// implement.  The merge keeps 0.1.59's generate.cpp, which calls them; the features behind them are
// opt-in and were not ported into the fork's chunk loop (see MERGE-NOTES.md).  run_multi declines with
// the exact error generate.cpp already handles as "read the prompt the usual way", so a batch co-read
// falls back to Prefill::run() instead of failing.
#include "strata/prefill/prefill.hpp"

#include <string>
#include <vector>

namespace strata::prefill {

void Prefill::set_ring_room_active(bool) {}   // STRATA_PREFILL_RING_ROOM: not ported (no ring room here)
bool Prefill::idle(std::string&) { return true; }
void Prefill::set_share_blocked(bool) {}      // STRATA_PREFILL_SHARE_BLOCKED: not ported
bool Prefill::nibble_prompt_check(std::string&) { return true; }   // no nibble prompt layout in this tree

bool Prefill::reserve_taps(const std::vector<int>&, int64_t, int64_t, std::string&) { return true; }   // DFlash2 taps: off

bool Prefill::run_multi(const std::vector<Seg>&, std::string& err) {
    err = "prefill: multi-prompt read declined: the 0.1.59 layer-major co-read is not ported into this engine";
    return false;
}

}  // namespace strata::prefill
