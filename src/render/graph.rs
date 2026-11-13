//! The audio graph topology and render algorithm

#[cfg(test)]
mod test;

use std::any::Any;
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};

use crate::context::AudioNodeId;
#[cfg(feature = "diagnostics")]
use crate::context::{AudioGraphDiagnostics, AudioGraphEdgeDiagnostics, AudioNodeDiagnostics};
use smallvec::{smallvec, SmallVec};

use super::node_collection::AudioNodeIdSet;
#[cfg(not(target_arch = "wasm32"))]
use super::BoundarySignal;
#[cfg(not(target_arch = "wasm32"))]
use super::ThreadAffinity;
use super::{Alloc, AudioParamValues, AudioProcessor, AudioRenderQuantum, NodeCollection};
use crate::node::{ChannelConfigInner, ChannelCountMode, ChannelInterpretation};
use crate::render::AudioWorkletGlobalScope;

/// Connection between two audio nodes
struct OutgoingEdge {
    /// index of the current Nodes output port
    self_index: usize,
    /// reference to the other Node
    other_id: AudioNodeId,
    /// index of the other Nodes input port
    other_index: usize,
}

impl std::fmt::Debug for OutgoingEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut format = f.debug_struct("OutgoingEdge");
        format
            .field("self_index", &self.self_index)
            .field("other_id", &self.other_id);
        if self.other_index == usize::MAX {
            format.field("other_index", &"HIDDEN");
        } else {
            format.field("other_index", &self.other_index);
        }
        format.finish()
    }
}

/// Renderer Node in the Audio Graph
pub struct Node {
    /// AudioNodeId, to be sent back to the control thread when this node is dropped
    reclaim_id: Option<llq::Node<AudioNodeId>>,
    /// Renderer: converts inputs to outputs
    processor: Box<dyn AudioProcessor>,
    /// Reusable input buffers
    inputs: Vec<AudioRenderQuantum>,
    /// Reusable output buffers, consumed by subsequent Nodes in this graph
    outputs: Vec<AudioRenderQuantum>,
    /// Channel configuration: determines up/down-mixing of inputs
    channel_config: ChannelConfigInner,
    /// Outgoing edges: tuple of outcoming node reference, our output index and their input index
    outgoing_edges: SmallVec<[OutgoingEdge; 2]>,
    /// Indicates if the control thread has dropped this Node
    control_handle_dropped: bool,
    /// Indicates if the node has any incoming connections (for lifecycle management)
    has_inputs_connected: bool,
    /// Indicates if the node can act as a cycle breaker (only DelayNode for now)
    cycle_breaker: bool,
    /// The nodes this cycle breaker was linked to before its outgoing edges were
    /// cleared to break a cycle (a DelayNode writer -> its reader). The two halves
    /// share processor state (the delay line) with no graph edge left between
    /// them, so the partition planner must keep them on the same render thread.
    cycle_break_partners: SmallVec<[AudioNodeId; 1]>,
    /// The processor is an `AudioParamProcessor` (set once, in `add_node`).
    is_audio_param: bool,
    /// AudioParam nodes evaluated inline, right before this node's own
    /// `process`, instead of as separate entries of the render order (see
    /// [`Graph::assign_inline_params`]). Recomputed on every re-ordering.
    inline_params: SmallVec<[AudioNodeId; 4]>,
    /// Every param in `inline_params` is settled: its last quantum started
    /// idle (empty timeline, silent input) and armed its idle fast path, so
    /// evaluating it again would only re-store the same `[[current value]]`
    /// and leave its output buffer as it is. While this holds, the owner skips
    /// its inline params altogether (one check per quantum instead of a borrow
    /// and a virtual `process` per param). Cleared whenever anything can
    /// change what one of them computes: a message routed to it (event,
    /// automation rate), a channel-config change, and every re-plan.
    params_clean: bool,
    /// For an inline param: the owner whose `inline_params` holds it, so a
    /// message to the param can clear the owner's `params_clean`.
    inline_owner: Option<AudioNodeId>,
}

/// Evaluate the AudioParams that `owner` reads, inline (see
/// [`Graph::assign_inline_params`]). Such a param has no incoming edge and its
/// only outgoing edge is the hidden one into its owner, so evaluating it right
/// before the owner is what the topological order does, minus the per-node
/// render-loop work that does nothing for it: no edge to push, nothing to
/// sum into or clear in its (silent, unconnected) input, and it cannot be
/// freed while it still feeds its owner.
///
/// Called inside the owner's `catch_unwind`: a panicking param takes its owner
/// down with it (the error is reported with the param's id). The owner would
/// fail on its next read of the vanished param anyway; this way the param is
/// handed back to the render order when its owner is freed and freed there.
///
/// Returns whether every param is now settled (see [`Node::params_clean`]).
#[inline]
fn process_inline_params(
    nodes: &NodeCollection,
    inline_params: &[AudioNodeId],
    scope: &AudioWorkletGlobalScope,
) -> bool {
    let mut settled = true;
    for &id in inline_params {
        let mut param = nodes.get_unchecked(id).borrow_mut();
        scope.node_id.set(id);
        param.process(AudioParamValues::from(nodes), scope);
        settled &= param.processor.param_settled();
        // An inline param has no incoming edge, but its input can still hold
        // residue summed in while it was an ordinary node that was not
        // rendered (muted in a cycle that is then broken). The render loop
        // consumes such residue once and clears it after `process`; do the
        // same, or the stale modulation is added again every quantum.
        if param.has_inputs_connected {
            param
                .inputs
                .iter_mut()
                .for_each(AudioRenderQuantum::make_silent);
            param.has_inputs_connected = false;
        }
    }
    settled
}

/// Evaluate `node`'s inline params unless they are all settled (see
/// [`Node::params_clean`]); restores `scope.node_id` to the owner after.
#[inline]
fn run_inline_params(
    nodes: &NodeCollection,
    node: &mut Node,
    index: AudioNodeId,
    params_clean_enabled: bool,
    scope: &AudioWorkletGlobalScope,
) {
    if node.params_clean || node.inline_params.is_empty() {
        return;
    }
    let settled = process_inline_params(nodes, &node.inline_params, scope);
    node.params_clean = settled && params_clean_enabled;
    scope.node_id.set(index);
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.reclaim_id.as_deref())
            .field("processor", &self.processor)
            .field("channel_config", &self.channel_config)
            .field("outgoing_edges", &self.outgoing_edges)
            .field("control_handle_dropped", &self.control_handle_dropped)
            .field("cycle_breaker", &self.cycle_breaker)
            .finish_non_exhaustive()
    }
}

impl Node {
    /// Render an audio quantum
    fn process(&mut self, params: AudioParamValues<'_>, scope: &AudioWorkletGlobalScope) -> bool {
        self.processor
            .process(&self.inputs[..], &mut self.outputs[..], params, scope)
    }

    /// Determine if this node is done playing and can be removed from the audio graph
    fn can_free(&self, tail_time: bool) -> bool {
        // Only drop when the Control thread has dropped its handle.
        // Otherwise the node can be reconnected/restarted etc.
        if !self.control_handle_dropped {
            return false;
        }

        // When the nodes has no incoming connections:
        if !self.has_inputs_connected {
            // Drop when the processor reports it won't yield output.
            if !tail_time {
                return true;
            }

            // Drop when the node does not have any inputs and outputs
            if self.outgoing_edges.is_empty() {
                return true;
            }
        }

        // Node has no control handle and does have inputs connected.
        // Drop when the processor when it has no outputs connected and does not have side effects
        if !self.processor.has_side_effects() && self.outgoing_edges.is_empty() {
            return true;
        }

        // Otherwise, do not drop the node.
        false
    }

    /// Get the current buffer for AudioParam values
    pub fn get_buffer(&self) -> &AudioRenderQuantum {
        self.outputs.first().unwrap()
    }
}

/// The audio graph
pub(crate) struct Graph {
    /// Processing Nodes
    nodes: NodeCollection,
    /// Allocator for audio buffers
    alloc: Alloc,
    /// Message channel to notify control thread of reclaimable AudioNodeIds
    reclaim_id_channel: llq::Producer<AudioNodeId>,
    /// Topological ordering of the nodes
    ordered: Vec<AudioNodeId>,
    /// Topological sorting helper
    marked: AudioNodeIdSet,
    /// Topological sorting helper
    marked_temp: Vec<AudioNodeId>,
    /// Topological sorting helper
    in_cycle: AudioNodeIdSet,
    /// Topological sorting helper
    cycle_breakers: Vec<AudioNodeId>,
    /// Cached partition plan for multicore rendering. Recomputed lazily whenever the
    /// topological ordering changes (i.e. whenever `ordered` is invalidated). `None`
    /// means "not yet computed" — the serial `render()` path never touches this.
    partition_plan: Option<PartitionPlan>,
    /// Opt-in: route rendering through the partitioned path ([`Graph::render_partitioned`])
    /// instead of serial [`Graph::render`]. Default `false`; on native builds it can be
    /// enabled with `WEB_AUDIO_RS_PARTITIONED=1`. The partitioned path is byte-identical,
    /// so this only affects *how* the quantum is computed, never the result bits.
    partitioned: bool,
    /// Number of worker threads for the multicore executor
    /// ([`Graph::render_partitioned_threaded`]). `0`/`1` disables threading (the
    /// serial partitioned or plain path is used instead). Set once at
    /// construction from `WEB_AUDIO_RS_PARALLEL` (native only). Byte-identical to
    /// serial regardless of worker count.
    parallel_workers: usize,
    /// Persistent pool of render worker threads, spawned once on the first
    /// threaded render and waiting for the next epoch between quanta. Rebuilt
    /// only when the worker count changes; torn down on drop. Spawning threads
    /// *per quantum* was measured to be a net loss (thread-create + join dwarfs
    /// the per-quantum DSP), so the threads live across the whole render.
    #[cfg(not(target_arch = "wasm32"))]
    pool: Option<WorkerPool>,
    /// Render epoch of the threaded executor, incremented once per quantum.
    #[cfg(not(target_arch = "wasm32"))]
    render_epoch: u64,
    /// Boundary slots of the threaded executor (one per cut edge), kept
    /// across quanta so their sample blocks are reused instead of allocated
    /// and freed on the render threads every quantum. Each signal carries the
    /// epoch it was written in; a merge only reads a slot of the current one.
    #[cfg(not(target_arch = "wasm32"))]
    boundary_store: Vec<Option<BoundarySignal>>,
    /// Measured partition costs of the last discarded plan, keyed by each
    /// partition's first node id, to seed the next plan's schedule.
    #[cfg(not(target_arch = "wasm32"))]
    partition_cost_memory: std::collections::HashMap<u64, u64>,
    /// Evaluate unconnected AudioParams inside their owner (default on; native
    /// builds can turn it off with `WEB_AUDIO_RS_INLINE_PARAMS=0` for A/B runs).
    /// Output is byte-identical either way.
    inline_params: bool,
    /// Skip the inline params of an owner while they are all settled (see
    /// [`Node::params_clean`]; default on; native builds can turn it off with
    /// `WEB_AUDIO_RS_PARAMS_CLEAN=0` for A/B runs). Output is byte-identical
    /// either way.
    params_clean: bool,
    /// Scratch for [`Graph::assign_inline_params`], indexed by `AudioNodeId.0`.
    inline_scratch: Vec<u8>,
    /// Inline params whose owner was freed this quantum; they rejoin `ordered`
    /// once the freed nodes are trimmed from it.
    orphaned_params: Vec<AudioNodeId>,
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("nodes", &self.nodes)
            .field("ordered", &self.ordered)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Partitioned (multicore) rendering plan
//
// The serial `render()` walks `ordered` once, and whenever it processes a node
// it immediately adds that node's output into each consumer's input. Because
// float addition is not associative, the exact bits of a fan-in (merge) node's
// input depend on the order its producers are processed — i.e. their position
// in `ordered`.
//
// To render independent sub-graphs on separate cores we must split the graph at
// its fan-in points, yet reproduce that exact fold order. The plan below does
// so by construction:
//
//   * A "merge node" is any node with >= 2 incoming *audio* edges. Every audio
//     edge into a merge node is "cut"; every other audio edge (into a node with
//     <= 1 audio input) is "direct". This guarantees the all-or-none invariant:
//     a merge node receives ALL its audio via cuts, a non-merge node receives
//     its single audio input directly.
//   * Partitions are the connected components under the *uncut* edges (direct
//     audio edges plus every AudioParam edge, so each param node stays with the
//     node it feeds). Each merge node roots its own component.
//   * A merge node's inputs are assembled by pulling its producers in `ordered`
//     order (see `MergeSlot::sources`) — the identical fold the serial path
//     performs, relocated from producer-time to merge-time. Partitions run in a
//     topological order of the partition DAG, so a merge node's producers have
//     always been processed (their outputs still valid) by the time it runs.
//
// Result: byte-identical output to `render()`, with partitions that carry no
// cross-partition data dependency except the cut edges summed at merge nodes.
// (This module computes and validates the plan and renders it *serially*; the
// thread-pool executor that runs partitions concurrently is a follow-up that
// reuses this exact plan and fold order.)
// ---------------------------------------------------------------------------

/// Assembly instructions for a single input port of a merge node: the ordered
/// list of boundary-buffer slots to sum, in global `ordered` order so the fold
/// is bit-identical to the serial render.
struct MergeSlot {
    input_index: usize,
    /// Indices into the per-quantum boundary-buffer table, in `ordered` order.
    sources: Vec<usize>,
}

/// All cut-input assembly instructions for one merge node (one entry per
/// connected input port).
struct MergeInputs {
    slots: Vec<MergeSlot>,
}

/// A precomputed plan to render the graph as independent partitions.
///
/// Each cut edge (producer -> merge node) owns a "boundary buffer" slot. When a
/// producer is processed, its output is *copied* into the boundary slots for its
/// cut edges (`copy_out`) — capturing the data before the producer can be freed,
/// and (in the threaded executor) decoupling it from the producer partition's
/// `Rc`-backed buffer pool. A merge node then sums its boundary slots in
/// `ordered` order (`merge_lookup`), reproducing the serial fold exactly.
struct PartitionPlan {
    /// Partitions in a valid execution order (every producer partition precedes
    /// the merge-node partitions it feeds). Each inner `Vec` lists a partition's
    /// node ids in global topological (`ordered`) order.
    partitions: Vec<Vec<AudioNodeId>>,
    /// Partition-DAG predecessors: `preds[p]` lists the partitions (indices into
    /// `partitions`, all smaller than `p`) whose cut-edge outputs partition `p`
    /// sums at its merge nodes. The threaded executor starts `p` once these are
    /// done in the current quantum.
    preds: Vec<Vec<usize>>,
    /// For each partition index, the worker thread it runs on. Empty until the
    /// threaded executor schedules the plan; the serial path ignores it.
    partition_worker: Vec<usize>,
    /// Per worker: the partitions it renders each quantum, in order (see
    /// [`Graph::schedule_partitions`]).
    worker_lists: Vec<Vec<usize>>,
    /// Number of workers the plan was scheduled for. `0` means "not yet
    /// scheduled".
    assigned_workers: usize,
    /// Quanta rendered since the last (re)schedule.
    quanta_since_schedule: usize,
    /// Per partition: smoothed render cost in nanoseconds, measured by the
    /// worker that runs it (one writer per quantum) and read by the scheduler
    /// between quanta. Seeded with an estimate from the node count.
    cost_ns: Vec<std::sync::atomic::AtomicU64>,
    /// Per partition: the render epoch in which it last completed. A consumer
    /// waits for `done[p] == epoch` of each predecessor `p`.
    done: Vec<std::sync::atomic::AtomicU64>,
    /// Dense, indexed by `AudioNodeId.0`: whether the node is a merge node (its
    /// audio inputs are assembled via `merge_lookup`, never pushed into directly).
    is_merge: Vec<bool>,
    /// Dense, indexed by `AudioNodeId.0`: for each node, the `(output_index,
    /// boundary_slot)` copies to perform right after it is processed. Empty for
    /// nodes with no cut outgoing edges.
    copy_out: Vec<Vec<(usize, usize)>>,
    /// Dense, indexed by `AudioNodeId.0`: the merge-assembly plan for each merge
    /// node (`None` for non-merge nodes).
    merge_lookup: Vec<Option<MergeInputs>>,
    /// Number of boundary-buffer slots to allocate per quantum.
    boundary_count: usize,
    /// False if the partition DAG could not be topologically ordered (e.g. an
    /// unexpected cycle); the caller then falls back to the serial `render()`.
    well_formed: bool,
}

/// Minimal union-find over node-id slots, used only when (re)computing a plan.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

// ---------------------------------------------------------------------------
// Multicore executor (native only)
// ---------------------------------------------------------------------------

/// Quanta between re-schedules from the measured partition costs.
#[cfg(not(target_arch = "wasm32"))]
const RESCHEDULE_QUANTA: usize = 64;

/// Measure partition costs on one quantum in this many (timing every partition
/// every quantum would add two clock reads per partition).
#[cfg(not(target_arch = "wasm32"))]
const COST_SAMPLE_QUANTA: u64 = 8;

/// Critical-path list scheduling of the partition DAG onto `n_workers`.
///
/// Repeatedly takes, among the partitions whose predecessors are all scheduled,
/// the one with the longest remaining path to the end of the DAG (its "bottom
/// level", in measured cost), and appends it to the worker that becomes free
/// earliest (worker 0 if `pinned`). Returns the worker of each partition and
/// each worker's list in execution order.
///
/// Deadlock freedom: a partition is appended only after all its predecessors
/// were appended (to any list), so every wait in a worker's list targets a
/// partition that appears earlier in this global append order; following the
/// waits always moves strictly backwards in that order, so no cycle of waits
/// can form.
#[cfg(not(target_arch = "wasm32"))]
fn list_schedule(
    costs: &[u64],
    preds: &[Vec<usize>],
    pinned: &[bool],
    n_workers: usize,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let n = costs.len();
    let n_workers = n_workers.max(1);
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut missing = vec![0usize; n];
    for (q, ps) in preds.iter().enumerate() {
        missing[q] = ps.len();
        for &p in ps {
            succs[p].push(q);
        }
    }
    // Predecessors always have smaller indices, so a reverse sweep sees every
    // successor's bottom level first.
    let mut bottom = vec![0u64; n];
    for p in (0..n).rev() {
        bottom[p] = costs[p] + succs[p].iter().map(|&s| bottom[s]).max().unwrap_or(0);
    }

    // Ties broken by the lower partition index, so the schedule is deterministic.
    let mut ready: std::collections::BinaryHeap<(u64, std::cmp::Reverse<usize>)> = (0..n)
        .filter(|&q| missing[q] == 0)
        .map(|q| (bottom[q], std::cmp::Reverse(q)))
        .collect();
    let mut ready_at = vec![0u64; n];
    let mut free_at = vec![0u64; n_workers];
    let mut partition_worker = vec![0usize; n];
    let mut worker_lists: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

    while let Some((_, std::cmp::Reverse(q))) = ready.pop() {
        let w = if pinned[q] {
            0
        } else {
            (0..n_workers).min_by_key(|&i| free_at[i]).unwrap()
        };
        let finish = free_at[w].max(ready_at[q]) + costs[q];
        free_at[w] = finish;
        partition_worker[q] = w;
        worker_lists[w].push(q);
        for &s in &succs[q] {
            ready_at[s] = ready_at[s].max(finish);
            missing[s] -= 1;
            if missing[s] == 0 {
                ready.push((bottom[s], std::cmp::Reverse(s)));
            }
        }
    }

    (partition_worker, worker_lists)
}

/// Per-quantum render task, published by the main thread into [`PoolShared`]
/// before releasing the workers. All fields are plain `Copy` data (raw pointers
/// and scalars); they point at render state that lives for the whole quantum:
/// `nodes` is a `Graph` field, while `plan`/`boundary` are
/// locals of [`Graph::render_partitioned_threaded`] that outlive the fork/join.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct PoolTask {
    nodes: *const NodeCollection,
    plan: *const PartitionPlan,
    boundary: *mut Option<BoundarySignal>,
    frame: u64,
    time: f64,
    sample_rate: f32,
    /// This quantum's render epoch (see [`PoolShared::epoch`]).
    epoch: u64,
    /// Whether to time the partitions this quantum (cost sampling).
    measure: bool,
    /// [`Graph::params_clean`].
    params_clean: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl PoolTask {
    const EMPTY: Self = PoolTask {
        nodes: std::ptr::null(),
        plan: std::ptr::null(),
        boundary: std::ptr::null_mut(),
        frame: 0,
        time: 0.0,
        sample_rate: 0.0,
        epoch: 0,
        measure: false,
        params_clean: false,
    };
}

/// State shared between the main thread and the persistent render workers.
///
/// # Safety (why the `unsafe impl Send + Sync` below are sound)
///
/// The raw pointers in `task` and the `!Send` render state they reach are only
/// touched between the start and the end of an epoch (one quantum):
///
/// * Start: the main thread writes `task`, then stores the new `epoch` with
///   Release. A worker reads `task` only after loading that epoch with Acquire,
///   so it sees the complete task.
/// * Dependencies: a partition's last action is storing the epoch into its
///   `done` mark (Release). A partition reads its producers' boundary slots only
///   after loading each producer's `done == epoch` (Acquire), so it sees the
///   complete boundary writes. Producers always come earlier in the global
///   schedule order (see [`list_schedule`]), so these waits cannot deadlock.
/// * End: a worker writes its `slots[w]`, then increments `finished` (Release);
///   the main thread loads `finished == n_workers - 1` (Acquire) before reading
///   any slot or touching the graph again.
/// * Disjointness: every node is touched by exactly one worker per quantum;
///   `AudioParam`/listener nodes are union-ed into the partition they feed, and
///   the two halves of a cycle-broken DelayNode (which share processor state
///   with no graph edge left between them) into one partition; each `boundary`
///   slot has one producer; each `slots[w]` is written only by worker `w`. So no
///   `RefCell<Node>`, shared processor state, boundary slot, or free-list is
///   ever touched by two threads at once.
/// * Buffers: render buffers are not tied to a pool. A worker allocates from
///   and releases to its own thread-local pool, so a buffer kept in processor
///   state that moves to another worker with its node after a re-plan just
///   joins that worker's pool. Buffer reference counts are atomic (`Arc`), as
///   two nodes' processor states may still share one buffer after a re-plan
///   splits them across workers.
#[cfg(not(target_arch = "wasm32"))]
struct PoolShared {
    /// The current render epoch; the main thread bumps it once per quantum to
    /// start the workers. Workers spin, then park, while it is unchanged.
    epoch: std::sync::atomic::AtomicU64,
    /// Spawned workers that finished the current epoch.
    finished: std::sync::atomic::AtomicUsize,
    /// Set on teardown; checked by each worker while waiting for an epoch.
    shutdown: std::sync::atomic::AtomicBool,
    /// The current quantum's task, published before the epoch is bumped.
    task: std::cell::UnsafeCell<PoolTask>,
    /// Per spawned worker (index `w - 1`): end-of-life node lists, filled before
    /// incrementing `finished`.
    slots: Vec<std::cell::UnsafeCell<Vec<AudioNodeId>>>,
}

// SAFETY: see the type-level doc — all shared access is barrier-ordered and
// index-disjoint, so concurrent reads/writes never alias.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for PoolShared {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for PoolShared {}

/// A persistent set of render worker threads, waiting for the next epoch between
/// quanta. Dropping the pool signals shutdown and joins every thread.
#[cfg(not(target_arch = "wasm32"))]
struct WorkerPool {
    n_workers: usize,
    shared: std::sync::Arc<PoolShared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Workers wait for an epoch between quanta (drop never races a render).
        // Flag shutdown, wake them, and join.
        self.shared
            .shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        for h in &self.handles {
            h.thread().unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WorkerPool {
    /// Spawn `n_workers - 1` persistent worker threads (the caller is worker 0).
    fn new(
        n_workers: usize,
        event_sender: &crossbeam_channel::Sender<crate::events::EventDispatch>,
    ) -> Self {
        let shared = std::sync::Arc::new(PoolShared {
            epoch: std::sync::atomic::AtomicU64::new(0),
            finished: std::sync::atomic::AtomicUsize::new(0),
            shutdown: std::sync::atomic::AtomicBool::new(false),
            task: std::cell::UnsafeCell::new(PoolTask::EMPTY),
            slots: (1..n_workers)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
        });

        let handles = (1..n_workers)
            .map(|w| {
                let shared = std::sync::Arc::clone(&shared);
                let sender = event_sender.clone();
                std::thread::Builder::new()
                    .name(format!("web-audio-render-{w}"))
                    .spawn(move || worker_thread_main(w, &shared, &sender))
                    .expect("spawn render worker")
            })
            .collect();

        WorkerPool {
            n_workers,
            shared,
            handles,
        }
    }
}

/// How long an idle worker spins for the next epoch before parking. Quanta of
/// an offline render, and the quanta within one audio callback, follow each
/// other closely, so a worker usually picks up the next epoch while spinning
/// and the OS wake-up of a park (tens of microseconds) is avoided; between
/// audio callbacks the workers park instead of burning a core.
#[cfg(not(target_arch = "wasm32"))]
const IDLE_SPIN: std::time::Duration = std::time::Duration::from_micros(100);

/// Spins on a dependency or completion condition before falling back to
/// yielding the core.
#[cfg(not(target_arch = "wasm32"))]
const DEPENDENCY_SPINS: u32 = 1 << 14;

/// Wait until `ready()` holds: spin first (dependencies usually resolve within
/// microseconds), then yield the core while waiting.
#[cfg(not(target_arch = "wasm32"))]
fn spin_then_yield(mut ready: impl FnMut() -> bool) {
    for _ in 0..DEPENDENCY_SPINS {
        if ready() {
            return;
        }
        std::hint::spin_loop();
    }
    while !ready() {
        std::thread::yield_now();
    }
}

/// Wait for an epoch other than `seen`, or `None` on shutdown. Spins for
/// [`IDLE_SPIN`], then parks; the main thread unparks every worker after
/// storing a new epoch, and a park after that unpark returns at once.
#[cfg(not(target_arch = "wasm32"))]
fn wait_for_epoch(shared: &PoolShared, seen: u64) -> Option<u64> {
    use std::sync::atomic::Ordering;
    let start = std::time::Instant::now();
    let mut spins = 0u32;
    loop {
        if shared.shutdown.load(Ordering::Acquire) {
            return None;
        }
        let epoch = shared.epoch.load(Ordering::Acquire);
        if epoch != seen {
            return Some(epoch);
        }
        spins = spins.wrapping_add(1);
        if spins % 64 != 0 || start.elapsed() < IDLE_SPIN {
            std::hint::spin_loop();
        } else {
            std::thread::park();
        }
    }
}

/// Entry point of a spawned render worker. Waits for each epoch, renders its
/// scheduled partitions for the published task, and signals completion, until
/// shutdown.
#[cfg(not(target_arch = "wasm32"))]
fn worker_thread_main(
    w: usize,
    shared: &PoolShared,
    sender: &crossbeam_channel::Sender<crate::events::EventDispatch>,
) {
    // Promote to a realtime scheduling class before entering the render loop so
    // the RT audio callback (worker 0) never waits on a normal-priority worker.
    // Best-effort; a failure only costs glitch margin. The guard lives until
    // this function returns, so the registration ends before the thread exits.
    let (priority, _promotion) = super::rt_thread::promote_current_thread();
    if priority != super::rt_thread::AudioThreadPriority::Promoted {
        log::debug!("render worker {w}: realtime priority not acquired ({priority:?})");
    }
    // Render with denormals flushed, exactly as worker 0 does (the online and
    // offline render paths run inside `no_denormals`): FTZ/DAZ live in a
    // per-thread register that spawned threads do not inherit, and a node must
    // see the same floating-point environment whichever worker runs it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    // SAFETY: as on worker 0 (see `RenderThread::render`): the flags are set for
    // the whole closure and restored when it returns.
    unsafe {
        no_denormals::no_denormals(|| worker_render_loop(w, shared, sender));
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    worker_render_loop(w, shared, sender);
}

/// The body of a render worker: wait for each epoch and render it, until
/// shutdown.
#[cfg(not(target_arch = "wasm32"))]
fn worker_render_loop(
    w: usize,
    shared: &PoolShared,
    sender: &crossbeam_channel::Sender<crate::events::EventDispatch>,
) {
    // Fill this worker's buffer pool before the first quantum.
    Alloc::prewarm(64);

    let mut seen = 0;
    while let Some(epoch) = wait_for_epoch(shared, seen) {
        seen = epoch;
        // Processor panics are caught per node; anything that unwinds out here
        // is an executor bug. The other threads would then wait forever for
        // this worker, so fail fast instead of hanging the render.
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: written before the epoch store we observed (see PoolShared).
            let task = unsafe { *shared.task.get() };
            // SAFETY: the pointed-at state lives for the whole quantum (see
            // PoolTask); this worker only reaches its own partitions' nodes.
            let nodes: &NodeCollection = unsafe { &*task.nodes };
            let plan: &PartitionPlan = unsafe { &*task.plan };
            let scope = AudioWorkletGlobalScope {
                current_frame: task.frame,
                current_time: task.time,
                sample_rate: task.sample_rate,
                node_id: std::cell::Cell::new(AudioNodeId(0)),
                event_sender: sender.clone(),
            };
            let freeable = run_worker_list(
                w,
                nodes,
                plan,
                task.boundary,
                &scope,
                task.epoch,
                task.measure,
                task.params_clean,
            );
            // SAFETY: only worker `w` writes `slots[w - 1]`; the main thread
            // reads it after the `finished` increment below.
            unsafe {
                *shared.slots[w - 1].get() = freeable;
            }
        }));
        if outcome.is_err() {
            log::error!("render worker {w} panicked outside a node; aborting");
            std::process::abort();
        }
        shared
            .finished
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

/// Render worker `w`'s scheduled partitions for one quantum, in its list order.
/// Each partition first waits until all its producers published `done` for this
/// epoch, then runs its nodes in serial order and publishes its own `done`.
/// Returns the end-of-life node ids it recorded.
#[cfg(not(target_arch = "wasm32"))]
fn run_worker_list(
    w: usize,
    nodes: &NodeCollection,
    plan: &PartitionPlan,
    boundary: *mut Option<BoundarySignal>,
    scope: &AudioWorkletGlobalScope,
    epoch: u64,
    measure: bool,
    params_clean: bool,
) -> Vec<AudioNodeId> {
    use std::sync::atomic::Ordering;
    let alloc = &Alloc;
    let mut freeable: Vec<AudioNodeId> = Vec::new();

    for &pidx in &plan.worker_lists[w] {
        for &producer in &plan.preds[pidx] {
            let done = &plan.done[producer];
            spin_then_yield(|| done.load(Ordering::Acquire) == epoch);
        }

        let started = measure.then(std::time::Instant::now);
        for &index in &plan.partitions[pidx] {
            process_node_threaded(
                nodes,
                plan,
                alloc,
                boundary,
                scope,
                index,
                epoch,
                params_clean,
                &mut freeable,
            );
        }
        if let Some(started) = started {
            // Exponential moving average (1/4 weight); this worker is the only
            // writer of this partition's cost in this epoch.
            let sample = started.elapsed().as_nanos() as u64;
            let cost = &plan.cost_ns[pidx];
            let old = cost.load(Ordering::Relaxed);
            cost.store(old - old / 4 + sample / 4, Ordering::Relaxed);
        }

        plan.done[pidx].store(epoch, Ordering::Release);
    }

    freeable
}

/// Process a single node on a worker thread. Mirrors the per-node body of
/// [`Graph::render_partitioned`] exactly, except cut-edge signals cross the
/// partition boundary as deep-copied [`BoundarySignal`]s and end-of-life nodes
/// are *recorded* (freed serially by the caller) rather than removed inline.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn process_node_threaded(
    nodes: &NodeCollection,
    plan: &PartitionPlan,
    alloc: &Alloc,
    boundary: *mut Option<BoundarySignal>,
    scope: &AudioWorkletGlobalScope,
    index: AudioNodeId,
    epoch: u64,
    params_clean: bool,
    freeable: &mut Vec<AudioNodeId>,
) {
    let node_idx = index.0 as usize;
    let mut node = nodes.get_unchecked(index).borrow_mut();

    // 1. If this is a merge node, assemble its inputs by summing the boundary
    //    buffers of its cut edges, in `ordered` order — the identical fold the
    //    serial path performs, rebuilt on this worker's allocator.
    if plan.is_merge[node_idx] {
        if let Some(merge) = &plan.merge_lookup[node_idx] {
            let channel_config = node.channel_config.clone();
            node.has_inputs_connected = true;
            for slot in &merge.slots {
                for &bidx in &slot.sources {
                    // SAFETY: filled by the producing partition, whose `done` for
                    // this epoch was observed (Acquire) before this partition
                    // started; this index is read-only here.
                    let signal = unsafe { (*boundary.add(bidx)).as_ref() }
                        .filter(|s| s.epoch == epoch)
                        .expect("boundary buffer filled before merge");
                    let quantum = alloc.boundary_to_quantum(signal);
                    node.inputs[slot.input_index].add(&quantum, &channel_config);
                }
            }
        }
    }

    // 2. Process (catch panics exactly as the serial paths do).
    let params = AudioParamValues::from(nodes);
    scope.node_id.set(index);
    let (success, tail_time) = {
        let catch_me = AssertUnwindSafe(|| {
            run_inline_params(nodes, &mut node, index, params_clean, scope);
            node.process(params, scope)
        });
        match panic::catch_unwind(catch_me) {
            Ok(tail_time) => (true, tail_time),
            Err(e) => {
                node.outgoing_edges.clear();
                scope.report_error(e);
                (false, false)
            }
        }
    };

    // 3. Copy this node's outputs into the boundary buffers of its cut edges.
    for &(output_index, bidx) in &plan.copy_out[node_idx] {
        // SAFETY: this worker is the sole producer for `bidx` (one cut edge ->
        // one slot); its consumer reads it only after this partition's `done`.
        unsafe {
            node.outputs[output_index].write_boundary(&mut *boundary.add(bidx), epoch);
        }
    }

    // 4. Push into the inputs of direct (non-merge) consumers. Such consumers
    //    are union-ed into this same partition, hence this same worker/allocator.
    node.outgoing_edges
        .iter()
        .filter(|edge| edge.other_index != usize::MAX)
        .filter(|edge| !plan.is_merge[edge.other_id.0 as usize])
        .for_each(|edge| {
            let mut output_node = nodes.get_unchecked(edge.other_id).borrow_mut();
            output_node.has_inputs_connected = true;
            let signal = &node.outputs[edge.self_index];
            let channel_config = &output_node.channel_config.clone();
            output_node.inputs[edge.other_index].add(signal, channel_config);
        });

    let can_free = !success || node.can_free(tail_time);

    if !can_free {
        node.inputs
            .iter_mut()
            .for_each(AudioRenderQuantum::make_silent);
        node.has_inputs_connected = false;
    } else {
        freeable.push(index);
    }

    drop(node);
}

impl Graph {
    pub fn new(reclaim_id_channel: llq::Producer<AudioNodeId>) -> Self {
        // Opt-in to the partitioned render path (native only). Read once at graph
        // construction so the hot render loop never touches the environment.
        #[cfg(not(target_arch = "wasm32"))]
        let partitioned =
            std::env::var("WEB_AUDIO_RS_PARTITIONED").is_ok_and(|v| v == "1" || v == "true");
        #[cfg(target_arch = "wasm32")]
        let partitioned = false;

        // Multicore executor opt-in (native only). `WEB_AUDIO_RS_PARALLEL=1|true`
        // uses all available cores; an explicit integer sets the worker count.
        #[cfg(not(target_arch = "wasm32"))]
        let parallel_workers = std::env::var("WEB_AUDIO_RS_PARALLEL").ok().map_or(0, |v| {
            if v == "1" || v == "true" {
                std::thread::available_parallelism().map_or(1, |n| n.get())
            } else {
                v.parse::<usize>().unwrap_or(0)
            }
        });
        #[cfg(target_arch = "wasm32")]
        let parallel_workers = 0;

        #[cfg(not(target_arch = "wasm32"))]
        let inline_params =
            !std::env::var("WEB_AUDIO_RS_INLINE_PARAMS").is_ok_and(|v| v == "0" || v == "false");
        #[cfg(target_arch = "wasm32")]
        let inline_params = true;
        #[cfg(test)]
        let inline_params = tests::INLINE_PARAMS_OVERRIDE
            .with(std::cell::Cell::get)
            .unwrap_or(inline_params);
        #[cfg(not(target_arch = "wasm32"))]
        let params_clean =
            !std::env::var("WEB_AUDIO_RS_PARAMS_CLEAN").is_ok_and(|v| v == "0" || v == "false");
        #[cfg(target_arch = "wasm32")]
        let params_clean = true;
        #[cfg(test)]
        let params_clean = tests::PARAMS_CLEAN_OVERRIDE
            .with(std::cell::Cell::get)
            .unwrap_or(params_clean);
        #[cfg(test)]
        let parallel_workers = tests::PARALLEL_OVERRIDE
            .with(std::cell::Cell::get)
            .unwrap_or(parallel_workers);

        Graph {
            nodes: NodeCollection::new(),
            alloc: Alloc::with_capacity(64),
            reclaim_id_channel,
            ordered: vec![],
            marked: AudioNodeIdSet::default(),
            marked_temp: vec![],
            in_cycle: AudioNodeIdSet::default(),
            cycle_breakers: vec![],
            partition_plan: None,
            partitioned,
            parallel_workers,
            #[cfg(not(target_arch = "wasm32"))]
            pool: None,
            #[cfg(not(target_arch = "wasm32"))]
            render_epoch: 0,
            #[cfg(not(target_arch = "wasm32"))]
            boundary_store: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            partition_cost_memory: std::collections::HashMap::new(),
            inline_params,
            params_clean,
            inline_scratch: Vec::new(),
            orphaned_params: Vec::new(),
        }
    }

    /// Override the number of multicore render-worker threads for this graph.
    ///
    /// Takes precedence over the `WEB_AUDIO_RS_PARALLEL` env var read in
    /// [`Graph::new`]. `0` or `1` render serially; `n >= 2` renders across `n`
    /// threads (byte-identical to serial). No-op in effect on wasm, where the
    /// parallel executor is not compiled in.
    pub(crate) fn set_parallel_workers(&mut self, workers: usize) {
        self.parallel_workers = workers;
    }

    /// Whether [`Graph::render_quantum`] renders on the worker pool.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn uses_worker_pool(&self) -> bool {
        self.parallel_workers >= 2
    }

    #[cfg(feature = "diagnostics")]
    pub fn diagnostics(&self) -> AudioGraphDiagnostics {
        let mut edge_count = 0;
        let mut cycle_breakers = Vec::new();
        let mut nodes = Vec::new();

        for id in self.nodes.keys() {
            let node = self.nodes.get_unchecked(id).borrow();
            let outgoing_edges: Vec<_> = node
                .outgoing_edges
                .iter()
                .map(|edge| AudioGraphEdgeDiagnostics {
                    output: edge.self_index,
                    destination: edge.other_id.0,
                    input: (edge.other_index != usize::MAX).then_some(edge.other_index),
                })
                .collect();

            edge_count += outgoing_edges.len();

            if node.cycle_breaker {
                cycle_breakers.push(id.0);
            }

            nodes.push(AudioNodeDiagnostics {
                id: id.0,
                processor: node.processor.name().to_string(),
                inputs: node.inputs.len(),
                outputs: node.outputs.len(),
                input_channels: node
                    .inputs
                    .iter()
                    .map(AudioRenderQuantum::number_of_channels)
                    .collect(),
                output_channels: node
                    .outputs
                    .iter()
                    .map(AudioRenderQuantum::number_of_channels)
                    .collect(),
                channel_config: format!("{:?}", node.channel_config),
                outgoing_edges,
                control_handle_dropped: node.control_handle_dropped,
                has_inputs_connected: node.has_inputs_connected,
                cycle_breaker: node.cycle_breaker,
                has_side_effects: node.processor.has_side_effects(),
            });
        }

        let node_count = nodes.len();

        AudioGraphDiagnostics {
            active: self.is_active(),
            node_count,
            edge_count,
            // the processing order: inline params right before their owner
            ordered: self
                .ordered
                .iter()
                .flat_map(|&id| {
                    let node = self.nodes.get_unchecked(id).borrow();
                    let mut ids: Vec<u64> = node.inline_params.iter().map(|p| p.0).collect();
                    ids.push(id.0);
                    ids
                })
                .collect(),
            in_cycle: self.in_cycle.iter().map(|id| id.0).collect(),
            cycle_breakers,
            nodes,
        }
    }

    /// Check if the graph is fully initialized and can start rendering
    pub fn is_active(&self) -> bool {
        // currently we only require the destination node to be present
        !self.nodes.is_empty()
    }

    pub fn add_node(
        &mut self,
        index: AudioNodeId,
        reclaim_id: llq::Node<AudioNodeId>,
        processor: Box<dyn AudioProcessor>,
        number_of_inputs: usize,
        number_of_outputs: usize,
        channel_config: ChannelConfigInner,
    ) {
        // todo: pre-allocate the buffers on the control thread

        // set input and output buffers to single channel of silence, will be upmixed when
        // necessary
        let inputs = vec![AudioRenderQuantum::from(self.alloc.silence()); number_of_inputs];
        let outputs = vec![AudioRenderQuantum::from(self.alloc.silence()); number_of_outputs];
        let is_audio_param =
            processor.name() == std::any::type_name::<crate::param::AudioParamProcessor>();

        self.nodes.insert(
            index,
            RefCell::new(Node {
                reclaim_id: Some(reclaim_id),
                processor,
                inputs,
                outputs,
                channel_config,
                outgoing_edges: smallvec![],
                control_handle_dropped: false,
                has_inputs_connected: false,
                cycle_breaker: false,
                cycle_break_partners: smallvec![],
                is_audio_param,
                inline_params: smallvec![],
                params_clean: false,
                inline_owner: None,
            }),
        );

        // add to ordered list (some nodes should be processed even if not connected)
        // if the node is connected later the ordered list will be recomputed
        self.ordered.push(index);
    }

    pub fn add_edge(&mut self, source: (AudioNodeId, usize), dest: (AudioNodeId, usize)) {
        self.nodes
            .get_unchecked_mut(source.0)
            .outgoing_edges
            .push(OutgoingEdge {
                self_index: source.1,
                other_id: dest.0,
                other_index: dest.1,
            });

        self.ordered.clear(); // void current ordering
        self.forget_partition_plan(); // topology changed, recompute partitions
    }

    pub fn remove_edge(&mut self, source: (AudioNodeId, usize), dest: (AudioNodeId, usize)) {
        self.nodes
            .get_unchecked_mut(source.0)
            .outgoing_edges
            .retain(|edge| {
                edge.other_id != dest.0 || edge.self_index != source.1 || edge.other_index != dest.1
            });

        // Removing an edge cannot invalidate an existing topological order. Re-sort only when
        // this removal may release nodes that were omitted because they are part of a cycle.
        if !self.in_cycle.is_empty() {
            self.ordered.clear();
        }
        // The partition plan, unlike `ordered`, encodes the exact edge set (which edges
        // are cut, their boundary slots, merge fold order), so *any* edge removal
        // invalidates it even when the topological order is still valid.
        self.forget_partition_plan();
    }

    pub fn mark_control_handle_dropped(&mut self, index: AudioNodeId) {
        // Issue #92, a race condition can occur for AudioParams. They may have already been
        // removed from the audio graph if the node they feed into was dropped.
        // Therefore, do not assume this node still exists:
        if let Some(node) = self.nodes.get_mut(index) {
            node.get_mut().control_handle_dropped = true;
        }
    }

    pub fn mark_cycle_breaker(&mut self, index: AudioNodeId) {
        self.nodes.get_unchecked_mut(index).cycle_breaker = true;
    }

    pub fn set_channel_count(&mut self, index: AudioNodeId, v: usize) {
        self.nodes.get_unchecked_mut(index).channel_config.count = v;
        self.mark_params_dirty(index);
    }
    pub fn set_channel_count_mode(&mut self, index: AudioNodeId, v: ChannelCountMode) {
        self.nodes
            .get_unchecked_mut(index)
            .channel_config
            .count_mode = v;
        self.mark_params_dirty(index);
    }
    pub fn set_channel_interpretation(&mut self, index: AudioNodeId, v: ChannelInterpretation) {
        self.nodes
            .get_unchecked_mut(index)
            .channel_config
            .interpretation = v;
        self.mark_params_dirty(index);
    }

    pub fn route_message(&mut self, index: AudioNodeId, msg: &mut dyn Any) {
        self.nodes.get_unchecked_mut(index).processor.onmessage(msg);
        self.mark_params_dirty(index);
    }

    /// Something may have changed what node `index` computes: if it is a param
    /// evaluated inline, its owner must evaluate its params again.
    fn mark_params_dirty(&mut self, index: AudioNodeId) {
        if let Some(owner) = self.nodes.get_unchecked_mut(index).inline_owner {
            self.nodes.get_unchecked_mut(owner).params_clean = false;
        }
    }

    /// Helper function for `order_nodes` - traverse node and outgoing edges
    ///
    /// The return value indicates `cycle_breaker_applied`:
    /// - true: a cycle was found and a cycle breaker was applied, current ordering is invalidated
    /// - false: visiting this leg was successful and no topological changes were applied
    fn visit(
        &self,
        node_id: AudioNodeId,
        marked: &mut AudioNodeIdSet,
        marked_temp: &mut Vec<AudioNodeId>,
        ordered: &mut Vec<AudioNodeId>,
        in_cycle: &mut AudioNodeIdSet,
        cycle_breakers: &mut Vec<AudioNodeId>,
    ) -> bool {
        // If this node is in the cycle detection list, it is part of a cycle!
        if let Some(pos) = marked_temp.iter().position(|&m| m == node_id) {
            // check if we can find some node that can break the cycle
            let cycle_breaker_node = marked_temp
                .iter()
                .skip(pos)
                .find(|&&node_id| self.nodes.get_unchecked(node_id).borrow().cycle_breaker);

            match cycle_breaker_node {
                Some(&node_id) => {
                    // store node id to clear the node outgoing edges
                    cycle_breakers.push(node_id);

                    return true;
                }
                None => {
                    // Mark all nodes in the cycle
                    in_cycle.extend(marked_temp[pos..].iter().copied());
                    // Do not continue, as we already have visited all these nodes
                    return false;
                }
            }
        }

        // Do not visit nodes multiple times
        if marked.contains(&node_id) {
            return false;
        }

        // Add node to the visited list
        marked.insert(node_id);
        // Add node to the current cycle detection list
        marked_temp.push(node_id);

        // Visit outgoing nodes, and call `visit` on them recursively
        for edge in self
            .nodes
            .get_unchecked(node_id)
            .borrow()
            .outgoing_edges
            .iter()
        {
            let cycle_breaker_applied = self.visit(
                edge.other_id,
                marked,
                marked_temp,
                ordered,
                in_cycle,
                cycle_breakers,
            );
            if cycle_breaker_applied {
                return true;
            }
        }

        // Then add this node to the ordered list
        ordered.push(node_id);

        // Finished visiting all nodes in this leg, clear the current cycle detection list
        marked_temp.retain(|marked| *marked != node_id);

        false
    }

    /// Determine the order of the audio nodes for rendering
    ///
    /// By inspecting the audio node connections, we can determine which nodes should render before
    /// other nodes. For example, in a graph with an audio source, a gain node and the destination
    /// node, at every render quantum the source should render first and after that the gain node.
    ///
    /// Inspired by the spec recommendation at
    /// <https://webaudio.github.io/web-audio-api/#rendering-loop>
    ///
    /// The goals are:
    /// - Perform a topological sort of the graph
    /// - Break cycles when possible (if there is a DelayNode present)
    /// - Mute nodes that are still in a cycle
    /// - For performance: no new allocations (reuse Vecs)
    fn order_nodes(&mut self) {
        // For borrowck reasons, we need the `visit` call to be &self.
        // So move out the bookkeeping Vecs, and pass them around as &mut.
        let mut ordered = std::mem::take(&mut self.ordered);
        let mut marked = std::mem::take(&mut self.marked);
        let mut marked_temp = std::mem::take(&mut self.marked_temp);
        let mut in_cycle = std::mem::take(&mut self.in_cycle);
        let mut cycle_breakers = std::mem::take(&mut self.cycle_breakers);

        // When a cycle breaker is applied, the graph topology changes and we need to run the
        // ordering again
        loop {
            // Clear previous administration
            ordered.clear();
            marked.clear();
            marked_temp.clear();
            in_cycle.clear();
            cycle_breakers.clear();

            // Visit all registered nodes, and perform a depth first traversal.
            //
            // We cannot just start from the AudioDestinationNode and visit all nodes connecting to it,
            // since the audio graph could contain legs detached from the destination and those should
            // still be rendered.
            let mut cycle_breaker_applied = false;
            for node_id in self.nodes.keys() {
                cycle_breaker_applied = self.visit(
                    node_id,
                    &mut marked,
                    &mut marked_temp,
                    &mut ordered,
                    &mut in_cycle,
                    &mut cycle_breakers,
                );

                if cycle_breaker_applied {
                    break;
                }
            }

            if cycle_breaker_applied {
                // clear the outgoing edges of the nodes that have been recognized as cycle breaker,
                // remembering who they fed so the partitioner can keep both halves together
                cycle_breakers.iter().for_each(|node_id| {
                    let node = self.nodes.get_unchecked_mut(*node_id);
                    for edge in node.outgoing_edges.drain(..) {
                        if !node.cycle_break_partners.contains(&edge.other_id) {
                            node.cycle_break_partners.push(edge.other_id);
                        }
                    }
                });

                continue;
            }

            break;
        }

        // Remove nodes from the ordering if they are part of a cycle. The spec mandates that their
        // outputs should be silenced, but with our rendering algorithm that is not necessary.
        // `retain` leaves the ordering in place
        ordered.retain(|o| !in_cycle.contains(o));

        // The `visit` function adds child nodes before their parent, so reverse the order
        ordered.reverse();

        // Re-instate Vecs
        self.ordered = ordered;
        self.marked = marked;
        self.marked_temp = marked_temp;
        self.in_cycle = in_cycle;
        self.cycle_breakers = cycle_breakers;

        self.assign_inline_params();
    }

    /// Move AudioParams out of the render order and into their owner's
    /// `inline_params`, where they are evaluated right before the owner.
    ///
    /// A param qualifies when nothing is connected into it (no incoming edge,
    /// so its input stays silent and its value depends only on its own
    /// timeline and the quantum's time) and its only outgoing edge is the
    /// hidden one into an owner that is itself rendered (not muted in a
    /// cycle). Evaluating it later in the quantum, right before the one node
    /// that reads it, then computes exactly the same values. A `connect()`
    /// into the param adds an edge, which re-orders the graph: from the next
    /// quantum on it is an ordinary node again, with its timeline state intact
    /// (the state lives in the processor, not in the graph), so the switch is
    /// on a quantum boundary and sample-exact.
    fn assign_inline_params(&mut self) {
        for node in self.nodes.values_mut() {
            let node = node.get_mut();
            node.inline_params.clear();
            node.params_clean = false;
            node.inline_owner = None;
        }
        if !self.inline_params {
            return;
        }

        const HAS_INCOMING: u8 = 1;
        const RENDERED: u8 = 2;
        let cap = self
            .nodes
            .keys()
            .map(|id| id.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let flags = &mut self.inline_scratch;
        flags.clear();
        flags.resize(cap, 0);
        for id in self.nodes.keys() {
            for edge in &self.nodes.get_unchecked(id).borrow().outgoing_edges {
                flags[edge.other_id.0 as usize] |= HAS_INCOMING;
            }
        }
        for id in &self.ordered {
            flags[id.0 as usize] |= RENDERED;
        }

        let nodes = &self.nodes;
        self.ordered.retain(|&id| {
            let node = nodes.get_unchecked(id).borrow();
            if !node.is_audio_param
                || flags[id.0 as usize] & HAS_INCOMING != 0
                || node.outgoing_edges.len() != 1
            {
                return true;
            }
            let edge = &node.outgoing_edges[0];
            let owner = edge.other_id;
            if edge.other_index != usize::MAX
                || owner == id
                || flags[owner.0 as usize] & RENDERED == 0
            {
                return true;
            }
            drop(node);
            nodes
                .get_unchecked(owner)
                .borrow_mut()
                .inline_params
                .push(id);
            nodes.get_unchecked(id).borrow_mut().inline_owner = Some(owner);
            false
        });
    }

    /// Hand the inline params of a freed node back to the render order (call
    /// after the freed nodes have been trimmed from `ordered`). They have no
    /// edges left, so any position is topologically valid; rendered there they
    /// are freed as usual once their control handle is gone.
    fn rehome_orphaned_params(&mut self) {
        if !self.orphaned_params.is_empty() {
            for &id in &self.orphaned_params {
                self.nodes.get_unchecked_mut(id).inline_owner = None;
            }
            self.ordered.append(&mut self.orphaned_params);
            self.forget_partition_plan();
        }
    }

    /// Render a single audio quantum, dispatching to the partitioned (multicore)
    /// path when enabled (see [`Graph::partitioned`]) or the serial path otherwise.
    /// Both produce byte-identical output. This is the entry point the render
    /// thread should call.
    pub fn render_quantum(&mut self, scope: &AudioWorkletGlobalScope) -> &AudioRenderQuantum {
        // Fill this (render) thread's buffer pool once, so it does not allocate
        // in its first quanta. A no-op after the first call on a thread.
        Alloc::prewarm(64);

        #[cfg(not(target_arch = "wasm32"))]
        if self.parallel_workers >= 2 {
            return self.render_partitioned_threaded(scope);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.partitioned {
            return self.render_partitioned(scope);
        }
        self.render(scope)
    }

    /// Render a single audio quantum by traversing the node list
    pub fn render(&mut self, scope: &AudioWorkletGlobalScope) -> &AudioRenderQuantum {
        // if the audio graph was changed, determine the new ordering
        if self.ordered.is_empty() {
            self.order_nodes();
        }

        // keep track of end-of-lifecyle nodes
        let mut nodes_dropped = false;

        // process every node, in topological sorted order
        self.ordered.iter().for_each(|index| {
            // acquire a mutable borrow of the current processing node
            let mut node = self.nodes.get_unchecked(*index).borrow_mut();

            // let the current node process (catch any panics that may occur)
            let params = AudioParamValues::from(&self.nodes);
            scope.node_id.set(*index);
            let (success, tail_time) = {
                // We are abusing AssertUnwindSafe here, we cannot guarantee it upholds.
                // This may lead to logic bugs later on, but it is the best that we can do.
                // The alternative is to crash and reboot the render thread.
                let catch_me = AssertUnwindSafe(|| {
                    run_inline_params(&self.nodes, &mut node, *index, self.params_clean, scope);
                    node.process(params, scope)
                });

                match panic::catch_unwind(catch_me) {
                    Ok(tail_time) => (true, tail_time),
                    Err(e) => {
                        node.outgoing_edges.clear();
                        scope.report_error(e);
                        (false, false)
                    }
                }
            };

            // iterate all outgoing edges, lookup these nodes and add to their input
            node.outgoing_edges
                .iter()
                // audio params are connected to the 'hidden' usize::MAX output, ignore them here
                .filter(|edge| edge.other_index != usize::MAX)
                .for_each(|edge| {
                    let mut output_node = self.nodes.get_unchecked(edge.other_id).borrow_mut();
                    output_node.has_inputs_connected = true;
                    let signal = &node.outputs[edge.self_index];
                    let channel_config = &output_node.channel_config.clone();

                    output_node.inputs[edge.other_index].add(signal, channel_config);
                });

            let can_free = !success || node.can_free(tail_time);

            // Node is not dropped.
            if !can_free {
                // Reset input buffers as they will be summed up in the next render quantum.
                node.inputs
                    .iter_mut()
                    .for_each(AudioRenderQuantum::make_silent);

                // Reset input state
                node.has_inputs_connected = false;
            }

            drop(node); // release borrow of self.nodes

            // Check if we can decommission this node (end of life)
            if can_free {
                // Node is dropped, remove it from the node list
                let mut node = self.nodes.remove(*index).into_inner();
                self.orphaned_params.extend(node.inline_params.drain(..));
                self.reclaim_id_channel
                    .push(node.reclaim_id.take().unwrap());
                node.processor.before_drop(scope);
                drop(node);

                // And remove it from the ordering after we have processed all nodes
                nodes_dropped = true;

                // Nodes are only dropped when they do not have incoming connections.
                // But they may have AudioParams feeding into them, these can de dropped too.
                self.nodes.values_mut().for_each(|node| {
                    // Check if this node was connected to the dropped node. In that case, it is
                    // either an AudioParam or the AudioListener that feeds into a PannerNode.
                    // These should be disconnected
                    let node = node.get_mut();
                    node.outgoing_edges.retain(|e| e.other_id != *index);
                    node.cycle_break_partners.retain(|p| p != index);
                });
            }
        });

        // If there were any nodes decommissioned, remove from graph order
        if nodes_dropped {
            let mut i = 0;
            while i < self.ordered.len() {
                if !self.nodes.contains(self.ordered[i]) {
                    self.ordered.remove(i);
                } else {
                    i += 1;
                }
            }
            self.rehome_orphaned_params();
        }

        // Return the output buffer of destination node
        &self.nodes.get_unchecked_mut(AudioNodeId(0)).outputs[0]
    }

    /// Compute the partition plan from the current `ordered` list and the graph
    /// topology. Pure (no mutation); callers cache the result in
    /// `self.partition_plan` and discard it whenever the topology changes.
    fn compute_partition_plan(&self) -> PartitionPlan {
        // Dense arrays are indexed by `AudioNodeId.0`; size for the largest id.
        let cap = self
            .nodes
            .keys()
            .map(|id| id.0 as usize)
            .max()
            .map_or(0, |m| m + 1);

        // 1. Count audio in-degree (param edges, other_index == usize::MAX, do
        //    not count). A "merge node" has >= 2 audio inputs.
        let mut audio_in_degree = vec![0usize; cap];
        for id in self.nodes.keys() {
            let node = self.nodes.get_unchecked(id).borrow();
            for edge in &node.outgoing_edges {
                if edge.other_index != usize::MAX {
                    audio_in_degree[edge.other_id.0 as usize] += 1;
                }
            }
        }
        // A node left out of `ordered` (muted in a cycle) is never processed,
        // but the serial render still sums its producers into its input, and
        // that residue is consumed once the cycle is broken. So it is never a
        // merge node: every producer pushes into it directly, which co-locates
        // all of them with it (step 2) and keeps the serial summation order.
        let mut in_ordered = vec![false; cap];
        for &id in &self.ordered {
            in_ordered[id.0 as usize] = true;
        }
        let mut shared_unrendered = false;
        let is_merge: Vec<bool> = audio_in_degree
            .iter()
            .zip(&in_ordered)
            .map(|(&d, &rendered)| {
                shared_unrendered |= d >= 2 && !rendered;
                d >= 2 && rendered
            })
            .collect();

        // 2. Union-find connected components under the *uncut* edges: every
        //    AudioParam edge (so a param node stays with the node it feeds) and
        //    every audio edge whose consumer is NOT a merge node. Audio edges
        //    into merge nodes are cut.
        let mut uf = UnionFind::new(cap);
        for id in self.nodes.keys() {
            let node = self.nodes.get_unchecked(id).borrow();
            let from = id.0 as usize;
            for edge in &node.outgoing_edges {
                let to = edge.other_id.0 as usize;
                if edge.other_index == usize::MAX {
                    uf.union(from, to); // param co-locates with its owner
                } else if !is_merge[to] {
                    uf.union(from, to); // single-input consumer stays with producer
                }
                // else: audio edge into a merge node -> cut (boundary)
            }
        }

        // 2b. Keep both halves of a cycle-broken DelayNode together. Breaking the
        //     cycle cleared the writer's edge to its reader, but the two still
        //     share the delay line (an `Rc<RefCell<..>>` plus the frame counters
        //     that order writer and reader), so they must run on the same thread,
        //     in serial order. Linking them can close a loop in the partition DAG
        //     (the feedback path through another partition back into the writer),
        //     so such loops are then collapsed into single partitions.
        let mut linked_partners = false;
        for id in self.nodes.keys() {
            let node = self.nodes.get_unchecked(id).borrow();
            for &partner in &node.cycle_break_partners {
                let p = partner.0 as usize;
                if p < cap && self.nodes.contains(partner) {
                    uf.union(id.0 as usize, p);
                    linked_partners = true;
                }
            }
        }
        // Co-locating the producers of an unrendered node can close loops too.
        if linked_partners || shared_unrendered {
            self.collapse_partition_cycles(&mut uf, &is_merge, cap);
        }

        // 3. Group nodes by component, in `ordered` sequence (so each partition's
        //    node list is already in global topological order).
        let mut group_of = vec![usize::MAX; cap]; // component root -> partition index
        let mut node_group = vec![usize::MAX; cap]; // node id -> partition index
        let mut partitions: Vec<Vec<AudioNodeId>> = Vec::new();
        for &id in &self.ordered {
            let root = uf.find(id.0 as usize);
            let gi = if group_of[root] == usize::MAX {
                let gi = partitions.len();
                group_of[root] = gi;
                partitions.push(Vec::new());
                gi
            } else {
                group_of[root]
            };
            node_group[id.0 as usize] = gi;
            partitions[gi].push(id);
        }
        let n_groups = partitions.len();

        // 4. Assign a boundary slot to each cut edge, in `ordered` producer order.
        //    Populate `copy_out` (producer-side copies) and `merge_lookup`
        //    (merge-side sum order). Iterating `self.ordered` guarantees each
        //    merge slot's `sources` list is in global topological order.
        let mut copy_out: Vec<Vec<(usize, usize)>> = vec![Vec::new(); cap];
        let mut merge_lookup: Vec<Option<MergeInputs>> = (0..cap).map(|_| None).collect();
        let mut boundary_count = 0usize;
        // Edges between partitions, for the partition DAG topo-sort (deduped).
        let mut dag_adj: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
        let mut dag_indeg = vec![0usize; n_groups];
        let mut seen_dag_edge: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();

        for &pid in &self.ordered {
            let node = self.nodes.get_unchecked(pid).borrow();
            let from_group = node_group[pid.0 as usize];
            for edge in &node.outgoing_edges {
                if edge.other_index == usize::MAX {
                    continue; // param edge
                }
                let to = edge.other_id.0 as usize;
                if node_group[to] == usize::MAX {
                    // Consumer is not in `ordered` (excluded as part of a cycle): a
                    // direct push (see `is_merge`), never a boundary slot or a
                    // partition-DAG edge.
                    continue;
                }
                if !is_merge[to] {
                    continue; // direct edge, pushed at producer-time (not a boundary)
                }
                // Cut edge: allocate a boundary slot.
                let slot = boundary_count;
                boundary_count += 1;
                copy_out[pid.0 as usize].push((edge.self_index, slot));

                let entry =
                    merge_lookup[to].get_or_insert_with(|| MergeInputs { slots: Vec::new() });
                match entry
                    .slots
                    .iter_mut()
                    .find(|s| s.input_index == edge.other_index)
                {
                    Some(s) => s.sources.push(slot),
                    None => entry.slots.push(MergeSlot {
                        input_index: edge.other_index,
                        sources: vec![slot],
                    }),
                }

                // Record the partition-DAG dependency from producer to merge.
                let to_group = node_group[to];
                if from_group != to_group && seen_dag_edge.insert((from_group, to_group)) {
                    dag_adj[from_group].push(to_group);
                    dag_indeg[to_group] += 1;
                }
            }
        }

        // 5. Topologically order the partitions (Kahn). Any valid topo order is
        //    correct for byte-identity; it only needs producers before merges.
        let mut queue: std::collections::VecDeque<usize> =
            (0..n_groups).filter(|&g| dag_indeg[g] == 0).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n_groups);
        while let Some(g) = queue.pop_front() {
            order.push(g);
            for &h in &dag_adj[g] {
                dag_indeg[h] -= 1;
                if dag_indeg[h] == 0 {
                    queue.push_back(h);
                }
            }
        }
        let well_formed = order.len() == n_groups;

        // Reorder partitions into execution order (if well-formed) and translate
        // the DAG edges into per-partition predecessor lists in execution indices.
        let mut ordered_partitions: Vec<Vec<AudioNodeId>> = Vec::new();
        let mut preds: Vec<Vec<usize>> = Vec::new();
        if well_formed {
            // group index -> execution index
            let mut exec_of_group = vec![usize::MAX; n_groups];
            for (exec_idx, &g) in order.iter().enumerate() {
                exec_of_group[g] = exec_idx;
                ordered_partitions.push(std::mem::take(&mut partitions[g]));
            }
            preds = vec![Vec::new(); n_groups];
            for (g, targets) in dag_adj.iter().enumerate() {
                for &h in targets {
                    preds[exec_of_group[h]].push(exec_of_group[g]);
                }
            }
        }

        // Seed costs with a node-count estimate until the executor measures them.
        let cost_ns = ordered_partitions
            .iter()
            .map(|p| std::sync::atomic::AtomicU64::new(1_000 * p.len() as u64))
            .collect();
        let done = ordered_partitions
            .iter()
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();

        PartitionPlan {
            partitions: ordered_partitions,
            preds,
            partition_worker: Vec::new(),
            worker_lists: Vec::new(),
            assigned_workers: 0,
            quanta_since_schedule: 0,
            cost_ns,
            done,
            is_merge,
            copy_out,
            merge_lookup,
            boundary_count,
            well_formed,
        }
    }

    /// Merge every strongly connected set of partitions (components of `uf`)
    /// into one partition, so the partition DAG built from `uf` is acyclic.
    ///
    /// Partition edges are the cut edges (audio edges into a merge node); all
    /// other edges are already inside a component. Only nodes in `ordered` take
    /// part, mirroring how [`Graph::compute_partition_plan`] builds the DAG.
    /// Iterative Kosaraju, so a deep graph cannot overflow the render thread's
    /// stack.
    fn collapse_partition_cycles(&self, uf: &mut UnionFind, is_merge: &[bool], cap: usize) {
        let mut in_ordered = vec![false; cap];
        for &id in &self.ordered {
            in_ordered[id.0 as usize] = true;
        }

        // Dense group ids for the components of ordered nodes, with one
        // representative node per group to union through.
        let mut group_of_root = vec![usize::MAX; cap];
        let mut rep: Vec<usize> = Vec::new();
        for &id in &self.ordered {
            let root = uf.find(id.0 as usize);
            if group_of_root[root] == usize::MAX {
                group_of_root[root] = rep.len();
                rep.push(id.0 as usize);
            }
        }
        let n = rep.len();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &id in &self.ordered {
            let from = group_of_root[uf.find(id.0 as usize)];
            let node = self.nodes.get_unchecked(id).borrow();
            for edge in &node.outgoing_edges {
                let to_node = edge.other_id.0 as usize;
                if edge.other_index == usize::MAX || !is_merge[to_node] || !in_ordered[to_node] {
                    continue; // param edge, intra-component edge, or excluded consumer
                }
                let to = group_of_root[uf.find(to_node)];
                if to != from {
                    adj[from].push(to);
                }
            }
        }

        // Pass 1: post-order of a DFS over `adj`.
        let mut visited = vec![false; n];
        let mut post: Vec<usize> = Vec::with_capacity(n);
        for start in 0..n {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
            while let Some(top) = stack.last_mut() {
                let v = top.0;
                if let Some(&w) = adj[v].get(top.1) {
                    top.1 += 1;
                    if !visited[w] {
                        visited[w] = true;
                        stack.push((w, 0));
                    }
                } else {
                    post.push(v);
                    stack.pop();
                }
            }
        }

        // Pass 2: components of the reversed graph, in reverse post-order.
        let mut radj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (v, targets) in adj.iter().enumerate() {
            for &w in targets {
                radj[w].push(v);
            }
        }
        let mut leader = vec![usize::MAX; n];
        for &start in post.iter().rev() {
            if leader[start] != usize::MAX {
                continue;
            }
            leader[start] = start;
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                for &w in &radj[v] {
                    if leader[w] == usize::MAX {
                        leader[w] = start;
                        stack.push(w);
                    }
                }
            }
        }

        for v in 0..n {
            if leader[v] != v {
                uf.union(rep[v], rep[leader[v]]);
            }
        }
    }

    /// Render a single audio quantum via the partition plan. Produces output that
    /// is byte-identical to [`Graph::render`]. This implementation runs partitions
    /// *serially* (in topological order); it is the correctness-equivalence proof
    /// that the threaded executor builds on. Native-only — the wasm/serial path
    /// stays on [`Graph::render`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_partitioned(&mut self, scope: &AudioWorkletGlobalScope) -> &AudioRenderQuantum {
        // if the audio graph was changed, determine the new ordering (and force a
        // plan recompute, since the topology — hence partitions — changed).
        if self.ordered.is_empty() {
            self.order_nodes();
            self.partition_plan = None;
        }
        if self.partition_plan.is_none() {
            self.partition_plan = Some(self.compute_partition_plan());
        }

        // Take the plan out so we can mutably borrow `self.nodes` during the pass.
        let plan = self.partition_plan.take().unwrap();

        if !plan.well_formed {
            // Unexpected: fall back to the serial path, but keep the (cached)
            // plan so we don't recompute it every quantum.
            self.partition_plan = Some(plan);
            return self.render(scope);
        }

        // Per-quantum boundary buffers. Filled by producers (as cheap `Rc` clones
        // of their outputs) before they can be freed; summed at merge nodes.
        let mut boundary: Vec<Option<AudioRenderQuantum>> =
            (0..plan.boundary_count).map(|_| None).collect();

        let mut nodes_dropped = false;

        for partition in &plan.partitions {
            for &index in partition {
                let node_idx = index.0 as usize;

                // acquire a mutable borrow of the current processing node
                let mut node = self.nodes.get_unchecked(index).borrow_mut();

                // 1. If this is a merge node, assemble its inputs by summing the
                //    boundary buffers of its cut edges, in `ordered` order — the
                //    identical fold `render()` performs incrementally.
                if plan.is_merge[node_idx] {
                    if let Some(merge) = &plan.merge_lookup[node_idx] {
                        let channel_config = node.channel_config.clone();
                        node.has_inputs_connected = true;
                        for slot in &merge.slots {
                            for &bidx in &slot.sources {
                                let signal = boundary[bidx]
                                    .as_ref()
                                    .expect("boundary buffer filled before merge");
                                node.inputs[slot.input_index].add(signal, &channel_config);
                            }
                        }
                    }
                }

                // 2. let the current node process (catch any panics that may occur)
                let params = AudioParamValues::from(&self.nodes);
                scope.node_id.set(index);
                let (success, tail_time) = {
                    let catch_me = AssertUnwindSafe(|| {
                        run_inline_params(&self.nodes, &mut node, index, self.params_clean, scope);
                        node.process(params, scope)
                    });
                    match panic::catch_unwind(catch_me) {
                        Ok(tail_time) => (true, tail_time),
                        Err(e) => {
                            node.outgoing_edges.clear();
                            scope.report_error(e);
                            (false, false)
                        }
                    }
                };

                // 3. Copy this node's outputs into the boundary buffers of its cut
                //    edges, capturing the data before the node can be freed.
                for &(output_index, bidx) in &plan.copy_out[node_idx] {
                    boundary[bidx] = Some(node.outputs[output_index].clone());
                }

                // 4. Push into the inputs of direct (non-merge) consumers, exactly
                //    as `render()` does for a node's outgoing edges.
                node.outgoing_edges
                    .iter()
                    .filter(|edge| edge.other_index != usize::MAX)
                    .filter(|edge| !plan.is_merge[edge.other_id.0 as usize])
                    .for_each(|edge| {
                        let mut output_node = self.nodes.get_unchecked(edge.other_id).borrow_mut();
                        output_node.has_inputs_connected = true;
                        let signal = &node.outputs[edge.self_index];
                        let channel_config = &output_node.channel_config.clone();
                        output_node.inputs[edge.other_index].add(signal, channel_config);
                    });

                let can_free = !success || node.can_free(tail_time);

                if !can_free {
                    node.inputs
                        .iter_mut()
                        .for_each(AudioRenderQuantum::make_silent);
                    node.has_inputs_connected = false;
                }

                drop(node); // release borrow of self.nodes

                if can_free {
                    let mut node = self.nodes.remove(index).into_inner();
                    self.orphaned_params.extend(node.inline_params.drain(..));
                    self.reclaim_id_channel
                        .push(node.reclaim_id.take().unwrap());
                    node.processor.before_drop(scope);
                    drop(node);

                    nodes_dropped = true;

                    self.nodes.values_mut().for_each(|node| {
                        let node = node.get_mut();
                        node.outgoing_edges.retain(|e| e.other_id != index);
                        node.cycle_break_partners.retain(|p| *p != index);
                    });
                }
            }
        }

        // If any nodes were decommissioned, trim them from the ordering and drop
        // the (now stale) partition plan so it is recomputed next quantum.
        if nodes_dropped {
            let mut i = 0;
            while i < self.ordered.len() {
                if !self.nodes.contains(self.ordered[i]) {
                    self.ordered.remove(i);
                } else {
                    i += 1;
                }
            }
            self.rehome_orphaned_params();
            // leave self.partition_plan == None -> recompute next quantum
        } else {
            self.partition_plan = Some(plan);
        }

        // Return the output buffer of destination node
        &self.nodes.get_unchecked_mut(AudioNodeId(0)).outputs[0]
    }

    /// (Re)compute which worker renders each partition, and in which order, from
    /// the measured partition costs (see [`list_schedule`]). Called when the plan
    /// is (re)computed, when the worker count changes, and every
    /// [`RESCHEDULE_QUANTA`] quanta so the schedule follows the measured costs.
    /// Only *where and when* a partition runs changes, never the per-node order or
    /// the merge fold order, so the output is byte-identical whatever the schedule.
    /// Render buffers need no re-homing: buffer pools are thread local, so a
    /// buffer that moves with a re-scheduled node simply joins its new worker's
    /// pool (see the `PoolShared` safety notes).
    #[cfg(not(target_arch = "wasm32"))]
    fn schedule_partitions(&self, plan: &mut PartitionPlan, n_workers: usize) {
        // Thread affinity: any partition containing a node that demands a stable
        // calling thread (`ThreadAffinity::RenderThread`, e.g. some CLAP plugins)
        // is pinned to worker 0 - the render origin thread, whose OS identity is
        // invariant across re-plans, re-schedules and pool resizes.
        let pinned: Vec<bool> = plan
            .partitions
            .iter()
            .map(|part| {
                part.iter().any(|&id| {
                    self.nodes
                        .get_unchecked(id)
                        .borrow()
                        .processor
                        .thread_affinity()
                        == ThreadAffinity::RenderThread
                })
            })
            .collect();
        let costs: Vec<u64> = plan
            .cost_ns
            .iter()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .collect();

        let (partition_worker, worker_lists) =
            list_schedule(&costs, &plan.preds, &pinned, n_workers);
        plan.partition_worker = partition_worker;
        plan.worker_lists = worker_lists;
        plan.assigned_workers = n_workers;
        plan.quanta_since_schedule = 0;
    }

    /// Render a single audio quantum across `parallel_workers` threads, producing
    /// output that is byte-identical to [`Graph::render`].
    ///
    /// Partitions (independent sub-graphs between fan-in points — see
    /// [`PartitionPlan`]) are statically pinned to worker threads and rendered a
    /// DAG-level at a time: all partitions in a level run concurrently, then a
    /// barrier, then the next level. Cross-partition (cut) edges are carried as
    /// deep-copied [`BoundarySignal`]s, so partitions never share a buffer
    /// through a cut edge. Merge nodes sum their boundary inputs in the
    /// exact `ordered` fold the serial path uses, so the result bits do not
    /// depend on the worker count. Native-only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_partitioned_threaded(
        &mut self,
        scope: &AudioWorkletGlobalScope,
    ) -> &AudioRenderQuantum {
        let n_workers = self.parallel_workers.max(2);

        if self.ordered.is_empty() {
            self.order_nodes();
            self.forget_partition_plan();
        }
        if self.partition_plan.is_none() {
            let mut plan = self.compute_partition_plan();
            if plan.well_formed {
                self.seed_partition_costs(&plan);
                self.schedule_partitions(&mut plan, n_workers);
            }
            self.partition_plan = Some(plan);
        }

        // Take the plan out so we can borrow `self.nodes` during the pass.
        let mut plan = self.partition_plan.take().unwrap();

        if !plan.well_formed {
            self.partition_plan = Some(plan);
            return self.render(scope);
        }

        // Follow the measured costs (and any change of the worker count).
        if plan.assigned_workers != n_workers || plan.quanta_since_schedule >= RESCHEDULE_QUANTA {
            self.schedule_partitions(&mut plan, n_workers);
        }
        plan.quanta_since_schedule += 1;

        // Boundary buffers (deep-copied cut-edge signals), persistent across
        // quanta (see `boundary_store`). Accessed from worker threads by raw
        // pointer at disjoint indices; a merge reads a slot only after the
        // producing partition published `done` for this epoch.
        if self.boundary_store.len() != plan.boundary_count {
            self.boundary_store.truncate(plan.boundary_count);
            self.boundary_store
                .resize_with(plan.boundary_count, || None);
        }
        let boundary_base = self.boundary_store.as_mut_ptr();

        // Ensure the persistent worker pool matches the current worker count.
        // Spawning per quantum was measured to be a net loss (thread create/join
        // dwarfs the per-quantum DSP), so the threads live across the render and
        // wait for the next epoch between quanta.
        if self.pool.as_ref().is_none_or(|p| p.n_workers != n_workers) {
            // Drop any stale pool first (joins its threads) before spawning anew.
            self.pool = None;
            self.pool = Some(WorkerPool::new(n_workers, &scope.event_sender));
        }
        self.render_epoch += 1;
        let epoch = self.render_epoch;
        let measure = epoch % COST_SAMPLE_QUANTA == 1;
        let pool = self.pool.as_ref().unwrap();
        let shared = &*pool.shared;

        // Publish this quantum's task, then start the epoch and wake the workers.
        // SAFETY: every worker finished the previous epoch (the `finished` wait
        // below) and reads `task` only after observing the new epoch, which is
        // stored with Release after this write.
        unsafe {
            *shared.task.get() = PoolTask {
                nodes: &self.nodes,
                plan: &plan,
                boundary: boundary_base,
                frame: scope.current_frame,
                time: scope.current_time,
                sample_rate: scope.sample_rate,
                epoch,
                measure,
                params_clean: self.params_clean,
            };
        }
        shared
            .finished
            .store(0, std::sync::atomic::Ordering::Relaxed);
        shared
            .epoch
            .store(epoch, std::sync::atomic::Ordering::Release);
        for handle in &pool.handles {
            handle.thread().unpark();
        }

        // The calling thread acts as worker 0.
        let scope0 = AudioWorkletGlobalScope {
            current_frame: scope.current_frame,
            current_time: scope.current_time,
            sample_rate: scope.sample_rate,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: scope.event_sender.clone(),
        };
        // As on the spawned workers: a panic escaping here would leave them
        // waiting for partitions that never complete, so fail fast.
        let own_freeable = match panic::catch_unwind(AssertUnwindSafe(|| {
            run_worker_list(
                0,
                &self.nodes,
                &plan,
                boundary_base,
                &scope0,
                epoch,
                measure,
                self.params_clean,
            )
        })) {
            Ok(freeable) => freeable,
            Err(_) => {
                log::error!("render worker 0 panicked outside a node; aborting");
                std::process::abort();
            }
        };

        // Wait for the spawned workers; each published its free-list before
        // incrementing `finished` (Release), so the Acquire load orders both
        // their slot writes and all their node/boundary accesses before ours.
        let spawned = n_workers - 1;
        spin_then_yield(|| shared.finished.load(std::sync::atomic::Ordering::Acquire) == spawned);
        let mut freeables: Vec<Vec<AudioNodeId>> = shared
            .slots
            .iter()
            .map(|s| {
                // SAFETY: ordered by the `finished` Acquire above; workers no
                // longer touch their slots until the next epoch.
                unsafe { std::mem::take(&mut *s.get()) }
            })
            .collect();
        freeables.push(own_freeable);

        // Join: decommission end-of-life nodes serially (mirrors `render()`), in
        // `ordered` sequence for deterministic reclaim ordering.
        let mut nodes_dropped = false;
        let to_free: AudioNodeIdSet = freeables.into_iter().flatten().collect();
        if !to_free.is_empty() {
            let ordered_snapshot = self.ordered.clone();
            for index in ordered_snapshot {
                if !to_free.contains(&index) {
                    continue;
                }
                let mut node = self.nodes.remove(index).into_inner();
                self.orphaned_params.extend(node.inline_params.drain(..));
                self.reclaim_id_channel
                    .push(node.reclaim_id.take().unwrap());
                scope.node_id.set(index);
                node.processor.before_drop(scope);
                drop(node);

                nodes_dropped = true;

                self.nodes.values_mut().for_each(|node| {
                    let node = node.get_mut();
                    node.outgoing_edges.retain(|e| e.other_id != index);
                    node.cycle_break_partners.retain(|p| *p != index);
                });
            }
        }

        if nodes_dropped {
            let mut i = 0;
            while i < self.ordered.len() {
                if !self.nodes.contains(self.ordered[i]) {
                    self.ordered.remove(i);
                } else {
                    i += 1;
                }
            }
            self.rehome_orphaned_params();
            // leave self.partition_plan == None -> recompute next quantum, seeded
            // with this plan's measured costs
            self.remember_partition_costs(&plan);
        } else {
            self.partition_plan = Some(plan);
        }

        // Return the output buffer of destination node
        &self.nodes.get_unchecked_mut(AudioNodeId(0)).outputs[0]
    }

    /// Drop the cached partition plan, keeping its measured partition costs to
    /// seed the next plan.
    fn forget_partition_plan(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(plan) = self.partition_plan.take() {
            self.remember_partition_costs(&plan);
        }
        self.partition_plan = None;
    }

    /// Save a plan's measured partition costs, keyed by each partition's first
    /// node id. Re-plans happen whenever a node is freed (e.g. every finished
    /// voice), and most partitions survive them unchanged; without this every
    /// re-plan would fall back to the node-count estimate.
    #[cfg(not(target_arch = "wasm32"))]
    fn remember_partition_costs(&mut self, plan: &PartitionPlan) {
        self.partition_cost_memory.clear();
        for (part, cost) in plan.partitions.iter().zip(&plan.cost_ns) {
            if let Some(first) = part.first() {
                self.partition_cost_memory
                    .insert(first.0, cost.load(std::sync::atomic::Ordering::Relaxed));
            }
        }
    }

    /// Seed a fresh plan's costs from the previous plan's measurements.
    #[cfg(not(target_arch = "wasm32"))]
    fn seed_partition_costs(&self, plan: &PartitionPlan) {
        for (part, cost) in plan.partitions.iter().zip(&plan.cost_ns) {
            if let Some(known) = part
                .first()
                .and_then(|first| self.partition_cost_memory.get(&first.0))
            {
                cost.store(*known, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    pub fn before_drop(&mut self, scope: &AudioWorkletGlobalScope) {
        self.nodes.iter_mut().for_each(|(id, node)| {
            scope.node_id.set(id);
            node.get_mut().processor.before_drop(scope);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::DESTINATION_NODE_ID;

    thread_local! {
        /// Test-only override of the inline-params switch for graphs built on
        /// this thread (an `OfflineAudioContext` builds its graph on the
        /// calling thread), so A/B tests need no process-wide env var.
        pub(super) static INLINE_PARAMS_OVERRIDE: std::cell::Cell<Option<bool>> =
            const { std::cell::Cell::new(None) };
        /// Test-only override of the params-clean switch, as above.
        pub(super) static PARAMS_CLEAN_OVERRIDE: std::cell::Cell<Option<bool>> =
            const { std::cell::Cell::new(None) };
        /// Test-only override of the worker count, as above.
        pub(super) static PARALLEL_OVERRIDE: std::cell::Cell<Option<usize>> =
            const { std::cell::Cell::new(None) };
    }

    /// Render an automation-heavy offline graph with params inline or as graph
    /// nodes: every automation kind, k-rate and a-rate, a param that gets an
    /// audio input connected and later disconnected mid-render, a param
    /// connected from the start, and an owner that ends (stop + dropped handle)
    /// while its params are inline.
    fn render_automation_graph(inline: bool, clean: bool, parallel: usize) -> Vec<Vec<f32>> {
        use crate::context::BaseAudioContext;
        use crate::context::OfflineAudioContext;
        use crate::node::{AudioNode, AudioScheduledSourceNode};
        use crate::AutomationRate;
        use std::sync::{Arc, Mutex};

        INLINE_PARAMS_OVERRIDE.with(|o| o.set(Some(inline)));
        PARAMS_CLEAN_OVERRIDE.with(|o| o.set(Some(clean)));
        PARALLEL_OVERRIDE.with(|o| o.set(Some(parallel)));
        let sr = 48000.;
        use crate::RENDER_QUANTUM_SIZE;
        let q = RENDER_QUANTUM_SIZE as f64 / sr as f64;
        let mut ctx = OfflineAudioContext::new(2, RENDER_QUANTUM_SIZE * 120, sr);
        INLINE_PARAMS_OVERRIDE.with(|o| o.set(None));
        PARAMS_CLEAN_OVERRIDE.with(|o| o.set(None));
        PARALLEL_OVERRIDE.with(|o| o.set(None));

        let mut osc = ctx.create_oscillator();
        osc.set_type(crate::node::OscillatorType::Sawtooth);
        osc.frequency().set_value_at_time(110., 0.);
        osc.frequency().linear_ramp_to_value_at_time(440., 7.3 * q);
        osc.frequency()
            .exponential_ramp_to_value_at_time(220., 19.6 * q);
        osc.detune().set_target_at_time(-300., 3.2 * q, 0.002);

        let filter = ctx.create_biquad_filter();
        filter.frequency().set_value(800.);
        filter.frequency().set_value_curve_at_time(
            &[300., 5000., 1200., 2500.],
            10.4 * q,
            12.1 * q,
        );
        filter.q().set_automation_rate(AutomationRate::K);
        filter.q().linear_ramp_to_value_at_time(8., 30.5 * q);
        filter.q().cancel_and_hold_at_time(25.25 * q);
        filter.gain().set_value(3.);

        let amp = ctx.create_gain();
        amp.gain().set_value_at_time(0.5, 2.5 * q);
        amp.gain().linear_ramp_to_value_at_time(1.0, 40.7 * q);
        amp.gain().cancel_scheduled_values(33.1 * q);

        let pan = ctx.create_stereo_panner();
        pan.pan().set_value(-0.3);

        // a modulator into the gain param, connected and disconnected mid-render
        let mut lfo = ctx.create_oscillator();
        lfo.frequency().set_value(7.);
        let lfo_gain = ctx.create_gain();
        lfo_gain.gain().set_value(0.25);
        lfo.connect(&lfo_gain);
        lfo.start();

        // a modulator connected from the start (param stays a graph node)
        let mut tremolo = ctx.create_constant_source();
        tremolo.offset().set_value_at_time(0.1, 12.0 * q);
        let delay = ctx.create_delay(1.);
        delay.delay_time().set_value(0.001);
        tremolo.connect(delay.delay_time());
        tremolo.start();

        // an owner that ends while its params are inline
        let mut blip = ctx.create_oscillator();
        blip.frequency().linear_ramp_to_value_at_time(990., 20. * q);
        blip.connect(&pan);
        blip.start();
        blip.stop_at(14.5 * q);
        drop(blip);

        osc.connect(&filter);
        filter.connect(&amp);
        amp.connect(&pan);
        pan.connect(&delay);
        delay.connect(&ctx.destination());
        osc.start();

        let amp = Arc::new(Mutex::new(amp));
        let lfo_gain = Arc::new(Mutex::new(lfo_gain));
        {
            let (amp, lfo_gain) = (Arc::clone(&amp), Arc::clone(&lfo_gain));
            ctx.suspend_sync(17. * q, move |_| {
                let amp = amp.lock().unwrap();
                lfo_gain.lock().unwrap().connect(amp.gain());
            });
        }
        {
            let lfo_gain = Arc::clone(&lfo_gain);
            ctx.suspend_sync(29. * q, move |_| {
                lfo_gain.lock().unwrap().disconnect();
            });
        }

        // Params idle (settled) for many quanta, then changed from the control
        // side: their owner must evaluate them again from the next quantum.
        let pan = Arc::new(Mutex::new(pan));
        let filter = Arc::new(Mutex::new(filter));
        let lfo = Arc::new(Mutex::new(lfo));
        {
            let pan = Arc::clone(&pan);
            ctx.suspend_sync(40. * q, move |_| {
                pan.lock().unwrap().pan().set_value(0.4);
            });
        }
        {
            let filter = Arc::clone(&filter);
            ctx.suspend_sync(44. * q, move |_| {
                let filter = filter.lock().unwrap();
                filter.gain().set_automation_rate(AutomationRate::K);
            });
        }
        {
            let filter = Arc::clone(&filter);
            // a future event, scheduled while the param is settled
            ctx.suspend_sync(45. * q, move |_| {
                let filter = filter.lock().unwrap();
                filter.gain().set_value_at_time(-6., 51.5 * q);
                filter.frequency().set_value(1800.);
            });
        }
        {
            let (lfo, amp) = (Arc::clone(&lfo), Arc::clone(&amp));
            ctx.suspend_sync(56. * q, move |_| {
                lfo.lock().unwrap().frequency().set_value(3.);
                amp.lock().unwrap().gain().set_value(0.75);
            });
        }
        {
            let pan = Arc::clone(&pan);
            ctx.suspend_sync(60. * q, move |_| {
                let pan = pan.lock().unwrap();
                pan.pan().linear_ramp_to_value_at_time(-0.8, 66.2 * q);
            });
        }

        let out = ctx.start_rendering_sync();
        (0..2).map(|c| out.get_channel_data(c).to_vec()).collect()
    }

    /// A GainNode whose gain param is fed by a constant source and by the gain
    /// node itself: a muted cycle. While muted, the param is not rendered but
    /// the source keeps summing into its input. After one quantum both inputs
    /// to the param are disconnected, so it becomes inline-eligible with that
    /// residue still in its input. Returns the first sample of each quantum.
    fn render_broken_param_cycle(inline: bool, parallel: usize) -> Vec<f32> {
        use crate::context::BaseAudioContext;
        use crate::context::OfflineAudioContext;
        use crate::node::{AudioNode, AudioScheduledSourceNode};
        use crate::RENDER_QUANTUM_SIZE;
        use std::sync::{Arc, Mutex};

        INLINE_PARAMS_OVERRIDE.with(|o| o.set(Some(inline)));
        PARALLEL_OVERRIDE.with(|o| o.set(Some(parallel)));
        let sr = 48000.;
        let q = RENDER_QUANTUM_SIZE as f64 / sr as f64;
        let mut ctx = OfflineAudioContext::new(1, RENDER_QUANTUM_SIZE * 6, sr);
        INLINE_PARAMS_OVERRIDE.with(|o| o.set(None));
        PARALLEL_OVERRIDE.with(|o| o.set(None));

        let mut src = ctx.create_constant_source();
        src.offset().set_value(1.);
        let gain = ctx.create_gain();
        src.connect(&gain);
        src.connect(gain.gain());
        gain.connect(gain.gain());
        gain.connect(&ctx.destination());
        src.start();

        let src = Arc::new(Mutex::new(src));
        let gain = Arc::new(Mutex::new(gain));
        ctx.suspend_sync(q, move |_| {
            let gain = gain.lock().unwrap();
            src.lock().unwrap().disconnect_dest(gain.gain());
            gain.disconnect_dest(gain.gain());
        });

        let out = ctx.start_rendering_sync();
        out.get_channel_data(0)
            .chunks(RENDER_QUANTUM_SIZE)
            .map(|c| c[0])
            .collect()
    }

    #[test]
    fn test_inline_param_drops_residue_of_broken_cycle() {
        let reference = render_broken_param_cycle(false, 0);
        // muted cycle, then the residue in the param input consumed once,
        // then the plain gain of 1
        assert_eq!(reference, [0., 4., 1., 1., 1., 1.]);
        for (inline, parallel) in [(true, 0), (false, 2), (true, 2), (true, 3)] {
            assert_eq!(
                render_broken_param_cycle(inline, parallel),
                reference,
                "inline = {inline}, parallel = {parallel}"
            );
        }
    }

    #[test]
    fn test_inline_params_byte_identical() {
        let reference = render_automation_graph(false, false, 0);
        assert!(reference[0].iter().any(|s| *s != 0.));
        for parallel in [0, 2, 3] {
            for (inline, clean) in [(false, true), (true, false), (true, true)] {
                let rendered = render_automation_graph(inline, clean, parallel);
                for (a, b) in reference.iter().zip(&rendered) {
                    let a: Vec<u32> = a.iter().map(|s| s.to_bits()).collect();
                    let b: Vec<u32> = b.iter().map(|s| s.to_bits()).collect();
                    assert_eq!(
                        a, b,
                        "inline = {inline}, clean = {clean}, parallel = {parallel}"
                    );
                }
            }
        }
    }

    /// A param processor that counts its `process` calls and reports itself
    /// settled, as an idle `AudioParamProcessor` does.
    #[derive(Debug)]
    struct CountingParam(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl AudioProcessor for CountingParam {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            _outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues<'_>,
            _scope: &AudioWorkletGlobalScope,
        ) -> bool {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            true
        }

        fn onmessage(&mut self, _msg: &mut dyn Any) {}

        fn param_settled(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_params_clean_skips_settled_params() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for clean in [true, false] {
            INLINE_PARAMS_OVERRIDE.with(|o| o.set(Some(true)));
            PARAMS_CLEAN_OVERRIDE.with(|o| o.set(Some(clean)));
            let mut graph = Graph::new(llq::Queue::new().split().0);
            INLINE_PARAMS_OVERRIDE.with(|o| o.set(None));
            PARAMS_CLEAN_OVERRIDE.with(|o| o.set(None));

            let node = Box::new(TestNode { tail_time: true });
            add_node(&mut graph, 0, node.clone());
            add_node(&mut graph, 1, node.clone());
            add_node(&mut graph, 2, node.clone());
            add_edge(&mut graph, 2, 0);
            let calls = std::sync::Arc::new(AtomicUsize::new(0));
            add_node(&mut graph, 3, Box::new(CountingParam(calls.clone())));
            graph.nodes.get_unchecked_mut(AudioNodeId(3)).is_audio_param = true;
            add_audioparam(&mut graph, 3, 2);

            let scope = AudioWorkletGlobalScope {
                current_frame: 0,
                current_time: 0.,
                sample_rate: 48000.,
                node_id: std::cell::Cell::new(AudioNodeId(0)),
                event_sender: crossbeam_channel::unbounded().0,
            };
            let mut expect = 0;
            let mut step = |graph: &mut Graph, evaluated: bool, why: &str| {
                graph.render(&scope);
                expect += usize::from(evaluated || !clean);
                let got = calls.load(Ordering::Relaxed);
                assert_eq!(got, expect, "{why}, clean = {clean}");
            };

            step(&mut graph, true, "first quantum");
            assert!(!graph.ordered.contains(&AudioNodeId(3)), "param is inline");
            step(&mut graph, false, "settled");
            step(&mut graph, false, "still settled");
            graph.route_message(AudioNodeId(3), &mut ());
            step(&mut graph, true, "message to the param");
            step(&mut graph, false, "settled again");
            graph.route_message(AudioNodeId(2), &mut ());
            step(
                &mut graph,
                false,
                "a message to the owner leaves its params",
            );
            graph.set_channel_count(AudioNodeId(3), 1);
            step(&mut graph, true, "channel config of the param");
            add_edge(&mut graph, 1, 0); // re-plan
            step(&mut graph, true, "re-plan");
            step(&mut graph, false, "settled after the re-plan");
        }
    }

    #[test]
    fn test_inline_audio_param_lifecycle() {
        let (node_id_producer, mut node_id_consumer) = llq::Queue::new().split();
        INLINE_PARAMS_OVERRIDE.with(|o| o.set(Some(true)));
        let mut graph = Graph::new(node_id_producer);
        INLINE_PARAMS_OVERRIDE.with(|o| o.set(None));

        let node = Box::new(TestNode { tail_time: false });
        add_node(&mut graph, 0, node.clone());
        add_node(&mut graph, 1, node.clone());
        // node 2 ends after one quantum; its param (3) is evaluated inline
        add_node(&mut graph, 2, node.clone());
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(2))
            .control_handle_dropped = true;
        add_edge(&mut graph, 2, 0);

        let param = Box::new(TestNode { tail_time: true });
        add_node(&mut graph, 3, param);
        let p = graph.nodes.get_unchecked_mut(AudioNodeId(3));
        p.control_handle_dropped = true;
        p.is_audio_param = true; // as `add_node` sets it for `AudioParamProcessor`
        add_audioparam(&mut graph, 3, 2);

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        graph.render(&scope); // node is dropped, its inline param rejoins the order
        assert!(!graph.ordered.contains(&AudioNodeId(2)));
        assert!(graph.ordered.contains(&AudioNodeId(3)));
        graph.render(&scope); // param is dropped

        assert_eq!(node_id_consumer.pop().unwrap().0, 2);
        assert_eq!(node_id_consumer.pop().unwrap().0, 3);
        assert!(node_id_consumer.pop().is_none());
        assert!(graph.ordered.iter().all(|id| id.0 < 2));
    }

    #[test]
    fn test_inline_params_assignment() {
        INLINE_PARAMS_OVERRIDE.with(|o| o.set(Some(true)));
        let mut graph = Graph::new(llq::Queue::new().split().0);
        INLINE_PARAMS_OVERRIDE.with(|o| o.set(None));
        let node = Box::new(TestNode { tail_time: true });
        for id in 0..6 {
            add_node(&mut graph, id, node.clone());
        }
        for id in 3..6 {
            graph
                .nodes
                .get_unchecked_mut(AudioNodeId(id))
                .is_audio_param = true;
        }
        add_edge(&mut graph, 2, 0);
        add_audioparam(&mut graph, 3, 2); // plain param: inline
        add_audioparam(&mut graph, 4, 2); // modulated param: stays a node
        add_edge(&mut graph, 1, 4);
        add_audioparam(&mut graph, 5, 2); // also feeds audio: stays a node
        add_edge(&mut graph, 5, 0);

        graph.order_nodes();
        assert!(!graph.ordered.contains(&AudioNodeId(3)));
        assert!(graph.ordered.contains(&AudioNodeId(4)));
        assert!(graph.ordered.contains(&AudioNodeId(5)));
        assert_eq!(
            graph
                .nodes
                .get_unchecked(AudioNodeId(2))
                .borrow()
                .inline_params[..],
            [AudioNodeId(3)]
        );

        // connecting into the inline param turns it back into a graph node
        add_edge(&mut graph, 1, 3);
        graph.order_nodes();
        assert!(graph.ordered.contains(&AudioNodeId(3)));
        assert!(graph
            .nodes
            .get_unchecked(AudioNodeId(2))
            .borrow()
            .inline_params
            .is_empty());
    }

    #[derive(Debug, Clone)]
    struct TestNode {
        tail_time: bool,
    }

    impl AudioProcessor for TestNode {
        fn process(
            &mut self,
            _inputs: &[AudioRenderQuantum],
            _outputs: &mut [AudioRenderQuantum],
            _params: AudioParamValues<'_>,
            _scope: &AudioWorkletGlobalScope,
        ) -> bool {
            self.tail_time
        }
    }

    fn config() -> ChannelConfigInner {
        ChannelConfigInner {
            count: 2,
            count_mode: crate::node::ChannelCountMode::Explicit,
            interpretation: crate::node::ChannelInterpretation::Speakers,
        }
    }

    fn add_node(graph: &mut Graph, id: u64, node: Box<dyn AudioProcessor>) {
        let id = AudioNodeId(id);
        let reclaim_id = llq::Node::new(id);
        graph.add_node(id, reclaim_id, node, 1, 1, config());
    }

    fn add_edge(graph: &mut Graph, from: u64, to: u64) {
        graph.add_edge((AudioNodeId(from), 0), (AudioNodeId(to), 0));
    }

    fn add_audioparam(graph: &mut Graph, from: u64, to: u64) {
        graph.add_edge((AudioNodeId(from), 0), (AudioNodeId(to), usize::MAX));
    }

    // regression test for:
    // https://github.com/orottier/web-audio-api-rs/issues/389
    #[test]
    fn test_active() {
        let mut graph = Graph::new(llq::Queue::new().split().0);
        assert!(!graph.is_active());
        // graph is active only when AudioDestination is set up
        let node = Box::new(TestNode { tail_time: false });
        add_node(&mut graph, DESTINATION_NODE_ID.0, node.clone());
        assert!(graph.is_active());
    }

    #[test]
    fn test_add_remove() {
        let mut graph = Graph::new(llq::Queue::new().split().0);

        let node = Box::new(TestNode { tail_time: false });
        add_node(&mut graph, 0, node.clone());
        add_node(&mut graph, 1, node.clone());
        add_node(&mut graph, 2, node.clone());
        add_node(&mut graph, 3, node);

        add_edge(&mut graph, 1, 0);
        add_edge(&mut graph, 2, 1);
        add_edge(&mut graph, 3, 0);

        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 4); // all nodes present
        assert_eq!(graph.ordered[3], AudioNodeId(0)); // root node comes last

        let pos1 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(1))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(2))
            .unwrap();
        assert!(pos2 < pos1); // node 1 depends on node 2

        // Detach node 1 (and thus node 2) from the root node
        graph.remove_edge((AudioNodeId(1), 0), (AudioNodeId(0), 0));
        graph.order_nodes();

        // sorting is not deterministic, but this should uphold:
        assert_eq!(graph.ordered.len(), 4); // all nodes present
        let pos1 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(1))
            .unwrap();
        let pos2 = graph
            .ordered
            .iter()
            .position(|&n| n == AudioNodeId(2))
            .unwrap();
        assert!(pos2 < pos1); // node 1 depends on node 2
    }

    #[test]
    fn remove_edge_preserves_acyclic_ordering() {
        let mut graph = Graph::new(llq::Queue::new().split().0);
        let node = Box::new(TestNode { tail_time: false });
        add_node(&mut graph, 0, node.clone());
        add_node(&mut graph, 1, node);
        add_edge(&mut graph, 1, 0);
        graph.order_nodes();

        let ordered = graph.ordered.clone();
        graph.remove_edge((AudioNodeId(1), 0), (AudioNodeId(0), 0));

        assert_eq!(graph.ordered, ordered);
    }

    #[test]
    fn remove_edge_invalidates_ordering_with_cycle() {
        let mut graph = Graph::new(llq::Queue::new().split().0);
        let node = Box::new(TestNode { tail_time: false });
        add_node(&mut graph, 0, node.clone());
        add_node(&mut graph, 1, node.clone());
        add_node(&mut graph, 2, node);
        add_edge(&mut graph, 1, 2);
        add_edge(&mut graph, 2, 1);
        graph.order_nodes();

        assert!(graph.in_cycle.contains(&AudioNodeId(1)));
        assert!(graph.in_cycle.contains(&AudioNodeId(2)));

        graph.remove_edge((AudioNodeId(2), 0), (AudioNodeId(1), 0));
        assert!(graph.ordered.is_empty());

        graph.order_nodes();
        assert!(graph.ordered.contains(&AudioNodeId(1)));
        assert!(graph.ordered.contains(&AudioNodeId(2)));
    }

    #[test]
    fn test_cycle() {
        let mut graph = Graph::new(llq::Queue::new().split().0);

        let node = Box::new(TestNode { tail_time: false });
        add_node(&mut graph, 0, node.clone());
        add_node(&mut graph, 1, node.clone());
        add_node(&mut graph, 2, node.clone());
        add_node(&mut graph, 3, node.clone());
        add_node(&mut graph, 4, node);

        // link 4->2, 2->1, 1->0, 1->2, 3->0
        add_edge(&mut graph, 4, 2);
        add_edge(&mut graph, 2, 1);
        add_edge(&mut graph, 1, 0);
        add_edge(&mut graph, 1, 2);
        add_edge(&mut graph, 3, 0);

        graph.order_nodes();

        let pos0 = graph.ordered.iter().position(|&n| n == AudioNodeId(0));
        let pos1 = graph.ordered.iter().position(|&n| n == AudioNodeId(1));
        let pos2 = graph.ordered.iter().position(|&n| n == AudioNodeId(2));
        let pos3 = graph.ordered.iter().position(|&n| n == AudioNodeId(3));
        let pos4 = graph.ordered.iter().position(|&n| n == AudioNodeId(4));

        // cycle 1<>2 should be removed
        assert_eq!(pos1, None);
        assert_eq!(pos2, None);
        // detached leg from cycle will still be rendered
        assert!(pos4.is_some());
        // a-cyclic part should be present
        assert!(pos3.unwrap() < pos0.unwrap());
    }

    #[test]
    fn test_lifecycle_and_reclaim() {
        let (node_id_producer, mut node_id_consumer) = llq::Queue::new().split();
        let mut graph = Graph::new(node_id_producer);

        let node = Box::new(TestNode { tail_time: false });

        // Destination Node is always node id 0, and should never drop
        add_node(&mut graph, 0, node.clone());

        // AudioListener Node is always node id 1, and should never drop
        add_node(&mut graph, 1, node.clone());

        // Add a regular node at id 3, it has tail time false so after rendering it should be
        // dropped and the AudioNodeId(3) should be reclaimed
        add_node(&mut graph, 2, node.clone());
        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(2))
            .control_handle_dropped = true;

        // Connect the regular node to the AudioDestinationNode
        add_edge(&mut graph, 2, 0);

        // Render a single quantum
        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };
        graph.render(&scope);

        // The dropped node should be our regular node, not the AudioListener
        let reclaimed = node_id_consumer
            .pop()
            .expect("should have decommisioned node");
        assert_eq!(reclaimed.0, 2);

        // No other dropped nodes
        assert!(node_id_consumer.pop().is_none());
    }

    #[test]
    fn test_audio_param_lifecycle() {
        let (node_id_producer, mut node_id_consumer) = llq::Queue::new().split();
        let mut graph = Graph::new(node_id_producer);

        let node = Box::new(TestNode { tail_time: false });

        // Destination Node is always node id 0, and should never drop
        add_node(&mut graph, 0, node.clone());

        // AudioListener Node is always node id 1, and should never drop
        add_node(&mut graph, 1, node.clone());

        // Add a regular node at id 3, it has tail time false so after rendering it should be
        // dropped and the AudioNodeId(3) should be reclaimed
        add_node(&mut graph, 2, node.clone());
        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(2))
            .control_handle_dropped = true;

        // Connect the regular node to the AudioDestinationNode
        add_edge(&mut graph, 2, 0);

        // Add an AudioParam at id 4, it should be dropped alongside the regular node
        let param = Box::new(TestNode { tail_time: true }); // audio params have tail time true
        add_node(&mut graph, 3, param);
        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(3))
            .control_handle_dropped = true;

        // Connect the audioparam to the regular node
        add_audioparam(&mut graph, 3, 2);

        // Render a single quantum
        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        // render twice
        graph.render(&scope); // node is dropped
        graph.render(&scope); // param is dropped

        // First the regular node should be dropped, then the audioparam
        assert_eq!(node_id_consumer.pop().unwrap().0, 2);
        assert_eq!(node_id_consumer.pop().unwrap().0, 3);

        // No other dropped nodes
        assert!(node_id_consumer.pop().is_none());
    }

    #[test]
    fn test_audio_param_with_signal_lifecycle() {
        let (node_id_producer, mut node_id_consumer) = llq::Queue::new().split();
        let mut graph = Graph::new(node_id_producer);

        let node = Box::new(TestNode { tail_time: false });

        // Destination Node is always node id 0, and should never drop
        add_node(&mut graph, 0, node.clone());

        // AudioListener Node is always node id 1, and should never drop
        add_node(&mut graph, 1, node.clone());

        // Add a regular node at id 3, it has tail time false so after rendering it should be
        // dropped and the AudioNodeId(3) should be reclaimed
        add_node(&mut graph, 2, node.clone());
        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(2))
            .control_handle_dropped = true;

        // Connect the regular node to the AudioDestinationNode
        add_edge(&mut graph, 2, 0);

        // Add an AudioParam at id 4, it should be dropped alongside the regular node
        let param = Box::new(TestNode { tail_time: true }); // audio params have tail time true
        add_node(&mut graph, 3, param);
        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(3))
            .control_handle_dropped = true;

        // Connect the audioparam to the regular node
        add_audioparam(&mut graph, 3, 2);

        // Add a source node to feed into the AudioParam
        let signal = Box::new(TestNode { tail_time: true });
        add_node(&mut graph, 4, signal);
        add_edge(&mut graph, 4, 3);
        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(4))
            .control_handle_dropped = true;

        // Render a single quantum
        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        // render twice
        graph.render(&scope); // node is dropped
        graph.render(&scope); // param is dropped

        // First the regular node should be dropped, then the audioparam
        assert_eq!(node_id_consumer.pop().unwrap().0, 2);
        assert_eq!(node_id_consumer.pop().unwrap().0, 3);

        // No other dropped nodes
        assert!(node_id_consumer.pop().is_none());

        // Render again
        graph.render(&scope); // param signal source is dropped
        assert_eq!(node_id_consumer.pop().unwrap().0, 4);
    }

    #[test]
    fn test_release_orphaned_source_nodes() {
        let (node_id_producer, mut node_id_consumer) = llq::Queue::new().split();
        let mut graph = Graph::new(node_id_producer);

        let node = Box::new(TestNode { tail_time: true });

        // Destination Node is always node id 0, and should never drop
        add_node(&mut graph, 0, node.clone());

        // AudioListener Node is always node id 1, and should never drop
        add_node(&mut graph, 1, node.clone());

        // Add a regular node at id 3, it has tail time true but since we drop the control handle
        // and there aren't any inputs and outputs, it will still be dropped and the AudioNodeId(3)
        // should be reclaimed
        add_node(&mut graph, 2, node);

        // Mark the node as 'detached from the control thread', so it is allowed to drop
        graph
            .nodes
            .get_unchecked_mut(AudioNodeId(2))
            .control_handle_dropped = true;

        // Render a single quantum
        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: std::cell::Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };
        graph.render(&scope);

        // The dropped node should be our orphaned node
        let reclaimed = node_id_consumer
            .pop()
            .expect("should have decommisioned node");
        assert_eq!(reclaimed.0, 2);

        // No other dropped nodes
        assert!(node_id_consumer.pop().is_none());
    }

    // The partitioned (multicore-plan) render path must be byte-identical to the
    // serial `render()`. We construct an order-sensitive fan-in (1e20, 1.0,
    // -1e20 all into one input) so that any deviation in the summation order
    // would change the result bits, then compare the two paths bit-for-bit.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_render_partitioned_byte_identical() {
        use std::cell::Cell;

        #[derive(Debug, Clone)]
        struct ConstSource {
            value: f32,
        }
        impl AudioProcessor for ConstSource {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true // keep alive (tail time) so nothing is freed mid-test
            }
        }

        #[derive(Debug, Clone)]
        struct PassThrough;
        impl AudioProcessor for PassThrough {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone(); // expose the summed input
                true
            }
        }

        fn build() -> Graph {
            let mut g = Graph::new(llq::Queue::new().split().0);
            // destination (id 0) is a merge node: 3 audio inputs summed into port 0
            add_node(&mut g, 0, Box::new(PassThrough));
            // listener placeholder (orphan, never dropped)
            add_node(&mut g, 1, Box::new(TestNode { tail_time: true }));
            // order-sensitive magnitudes: the fold result depends on summation order
            add_node(&mut g, 2, Box::new(ConstSource { value: 1e20 }));
            add_node(&mut g, 3, Box::new(ConstSource { value: 1.0 }));
            add_node(&mut g, 4, Box::new(ConstSource { value: -1e20 }));
            add_edge(&mut g, 2, 0);
            add_edge(&mut g, 3, 0);
            add_edge(&mut g, 4, 0);
            g
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        let mut serial = build();
        let mut parted = build();

        let a: Vec<u32> = serial
            .render(&scope)
            .channel_data(0)
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let b: Vec<u32> = parted
            .render_partitioned(&scope)
            .channel_data(0)
            .iter()
            .map(|x| x.to_bits())
            .collect();

        assert_eq!(a, b, "partitioned render must be byte-identical to serial");

        // Assert the partitioned path actually exercised the merge machinery
        // (not a trivial single-partition / fallback case).
        let plan = parted
            .partition_plan
            .as_ref()
            .expect("plan cached (no nodes dropped)");
        assert!(plan.well_formed);
        assert!(plan.is_merge[0], "destination should be a merge node");
        assert_eq!(
            plan.boundary_count, 3,
            "three cut edges -> three boundaries"
        );
    }

    // The multicore (threaded) render path must be byte-identical to the serial
    // `render()` for any worker count. We build several independent chains that
    // fan in to an order-sensitive merge, pin them across worker threads, and
    // compare bit-for-bit over multiple quanta.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_render_partitioned_threaded_byte_identical() {
        use std::cell::Cell;

        #[derive(Debug, Clone)]
        struct ConstSource {
            value: f32,
        }
        impl AudioProcessor for ConstSource {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true
            }
        }

        // A unity-gain pass-through that also does an in-place arithmetic op, so
        // each chain allocates/mutates buffers on its own worker's pool.
        #[derive(Debug, Clone)]
        struct Scale {
            k: f32,
        }
        impl AudioProcessor for Scale {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone();
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s *= self.k);
                true
            }
        }

        #[derive(Debug, Clone)]
        struct PassThrough;
        impl AudioProcessor for PassThrough {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone();
                true
            }
        }

        // Six fan-in chains (source -> scale -> destination merge). Large,
        // order-sensitive magnitudes make the fold order observable in the bits.
        let values: [f32; 6] = [1e20, 1.0, -1e20, 3.5, -2.5, 7.0];
        fn build(values: &[f32; 6]) -> Graph {
            let mut g = Graph::new(llq::Queue::new().split().0);
            add_node(&mut g, 0, Box::new(PassThrough)); // destination (merge)
            add_node(&mut g, 1, Box::new(TestNode { tail_time: true })); // listener
            for (i, &v) in values.iter().enumerate() {
                let src = 2 + (i as u64) * 2;
                let scale = src + 1;
                add_node(&mut g, src, Box::new(ConstSource { value: v }));
                add_node(&mut g, scale, Box::new(Scale { k: 1.0 }));
                add_edge(&mut g, src, scale); // direct edge (same partition)
                add_edge(&mut g, scale, 0); // cut edge into the merge
            }
            g
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        // Serial reference output (a few quanta).
        let serial_bits: Vec<Vec<u32>> = {
            let mut g = build(&values);
            (0..4)
                .map(|_| {
                    g.render(&scope)
                        .channel_data(0)
                        .iter()
                        .map(|x| x.to_bits())
                        .collect()
                })
                .collect()
        };

        // Threaded output must match for every worker count we try.
        for workers in [2usize, 3, 4, 8] {
            let mut g = build(&values);
            g.parallel_workers = workers;
            for (q, expected) in serial_bits.iter().enumerate() {
                let got: Vec<u32> = g
                    .render_partitioned_threaded(&scope)
                    .channel_data(0)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect();
                assert_eq!(
                    &got, expected,
                    "threaded render (workers={workers}, quantum={q}) must be byte-identical to serial"
                );
            }

            // Confirm the plan really partitioned and spread across workers.
            let plan = g
                .partition_plan
                .as_ref()
                .expect("plan cached (no nodes dropped)");
            assert!(plan.well_formed);
            assert!(plan.is_merge[0], "destination should be a merge node");
            assert_eq!(plan.boundary_count, 6, "six cut edges -> six boundaries");
            let used_workers = plan
                .partition_worker
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len();
            assert!(
                used_workers >= workers.min(6),
                "partitions should spread across available workers"
            );
        }
    }

    // A node reporting `ThreadAffinity::RenderThread` must always run on worker 0
    // — the render origin (calling) thread — even as the graph is edited and the
    // partition plan is recomputed between quanta. Without the pin, a replan's
    // round-robin could migrate its partition onto a spawned worker, tripping the
    // foreign-code thread assertion it stands in for (e.g. a CLAP plugin).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_thread_affinity_pins_to_render_thread_across_replans() {
        use crate::render::ThreadAffinity;
        use std::cell::Cell;
        use std::sync::{Arc, Mutex};
        use std::thread::ThreadId;

        #[derive(Debug, Clone)]
        struct ConstSource {
            value: f32,
        }
        impl AudioProcessor for ConstSource {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true
            }
        }

        #[derive(Debug, Clone)]
        struct PassThrough;
        impl AudioProcessor for PassThrough {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone();
                true
            }
        }

        // A source that records the OS thread it is processed on each quantum and
        // demands to always run on the render origin thread.
        #[derive(Clone)]
        struct Pinned {
            value: f32,
            threads: Arc<Mutex<Vec<ThreadId>>>,
        }
        impl std::fmt::Debug for Pinned {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Pinned")
            }
        }
        impl AudioProcessor for Pinned {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                self.threads
                    .lock()
                    .unwrap()
                    .push(std::thread::current().id());
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true
            }
            fn thread_affinity(&self) -> ThreadAffinity {
                ThreadAffinity::RenderThread
            }
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        const PINNED_ID: u64 = 2;
        let threads = Arc::new(Mutex::new(Vec::new()));

        // Destination merge (0) + a listener (1) to keep the graph alive, a
        // pinned source (2), and four plain sources — each a cut edge into the
        // merge, so the layer-0 partitions spread across workers.
        let mut g = Graph::new(llq::Queue::new().split().0);
        g.parallel_workers = 4;
        add_node(&mut g, 0, Box::new(PassThrough));
        add_node(&mut g, 1, Box::new(TestNode { tail_time: true }));
        add_node(
            &mut g,
            PINNED_ID,
            Box::new(Pinned {
                value: 0.25,
                threads: Arc::clone(&threads),
            }),
        );
        add_edge(&mut g, PINNED_ID, 0);
        for i in 0..4u64 {
            let src = 10 + i;
            add_node(&mut g, src, Box::new(ConstSource { value: i as f32 }));
            add_edge(&mut g, src, 0);
        }

        // worker 0 runs on *this* thread (the caller acts as worker 0).
        let origin = std::thread::current().id();

        // Render, forcing a partition replan before each pass by editing the
        // graph (adding a fresh source chain → `partition_plan = None`).
        for pass in 0..4u64 {
            if pass > 0 {
                let src = 100 + pass;
                add_node(&mut g, src, Box::new(ConstSource { value: pass as f32 }));
                add_edge(&mut g, src, 0);
            }
            threads.lock().unwrap().clear();

            let _ = g.render_partitioned_threaded(&scope);

            let plan = g
                .partition_plan
                .as_ref()
                .expect("plan cached (no nodes dropped)");
            assert!(plan.well_formed);

            // The pinned node's partition must be assigned to worker 0.
            let pinned_part = plan
                .partitions
                .iter()
                .position(|p| p.contains(&AudioNodeId(PINNED_ID)))
                .expect("pinned node must live in some partition");
            assert_eq!(
                plan.partition_worker[pinned_part], 0,
                "pass {pass}: pinned partition must be assigned to worker 0"
            );

            // Genuinely multi-worker, so the pin is doing real work.
            let used = plan
                .partition_worker
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len();
            assert!(used > 1, "pass {pass}: graph should span multiple workers");

            // And the processor was in fact only ever called on worker 0's
            // (this) thread.
            let recorded = threads.lock().unwrap();
            assert!(!recorded.is_empty(), "pass {pass}: pinned node must run");
            for &tid in recorded.iter() {
                assert_eq!(
                    tid, origin,
                    "pass {pass}: pinned node ran on a non-origin thread"
                );
            }
        }
    }

    // A live graph edit between quanta must invalidate the cached partition plan,
    // and the threaded executor must then render the EDITED topology
    // byte-identically to the serial render of that same edited graph. Guards the
    // "graph edits mid-render" path: a stale plan or a stale worker assignment
    // would leak the pre-edit graph's output.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_threaded_plan_invalidated_by_graph_edit() {
        use std::cell::Cell;

        #[derive(Debug, Clone)]
        struct ConstSource {
            value: f32,
        }
        impl AudioProcessor for ConstSource {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true
            }
        }

        #[derive(Debug, Clone)]
        struct PassThrough;
        impl AudioProcessor for PassThrough {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone();
                true
            }
        }

        // Five order-sensitive sources fan into a merge (0), with a listener (1)
        // to keep the graph alive. Source id = 2 + index.
        fn build(values: &[f32]) -> Graph {
            let mut g = Graph::new(llq::Queue::new().split().0);
            add_node(&mut g, 0, Box::new(PassThrough));
            add_node(&mut g, 1, Box::new(TestNode { tail_time: true }));
            for (i, &v) in values.iter().enumerate() {
                let src = 2 + i as u64;
                add_node(&mut g, src, Box::new(ConstSource { value: v }));
                add_edge(&mut g, src, 0);
            }
            g
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        let values: [f32; 5] = [1e20, 1.0, -1e20, 3.5, 7.0];
        let mut threaded = build(&values);
        threaded.parallel_workers = 4;
        let mut serial = build(&values);

        // Quantum 0: warm the threaded executor so a plan is cached.
        let _ = threaded.render_partitioned_threaded(&scope);
        assert_eq!(
            threaded.partition_plan.as_ref().unwrap().boundary_count,
            5,
            "five cut edges -> five boundaries before the edit"
        );

        // Identical edit on both graphs: drop the last source's cut edge (id 6).
        let last = AudioNodeId(2 + (values.len() as u64 - 1));
        threaded.remove_edge((last, 0), (AudioNodeId(0), 0));
        serial.remove_edge((last, 0), (AudioNodeId(0), 0));
        assert!(
            threaded.partition_plan.is_none(),
            "a graph edit must invalidate the cached partition plan"
        );

        // Quantum 1: the threaded render of the edited graph must match the
        // serial render of the identically-edited graph, bit for bit.
        let got: Vec<u32> = threaded
            .render_partitioned_threaded(&scope)
            .channel_data(0)
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let expected: Vec<u32> = serial
            .render(&scope)
            .channel_data(0)
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(
            got, expected,
            "threaded render after a graph edit must equal serial of the edited topology"
        );

        // The plan was recomputed with one fewer boundary (no stale boundary_count).
        assert_eq!(
            threaded.partition_plan.as_ref().unwrap().boundary_count,
            4,
            "removed cut edge -> one fewer boundary after replan"
        );
    }

    // With far more workers than useful partitions (and typically more than
    // cores), the threaded executor must still terminate (no barrier deadlock
    // with many idle workers) and stay byte-identical to serial.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_threaded_more_workers_than_partitions() {
        use std::cell::Cell;

        #[derive(Debug, Clone)]
        struct ConstSource {
            value: f32,
        }
        impl AudioProcessor for ConstSource {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true
            }
        }

        #[derive(Debug, Clone)]
        struct PassThrough;
        impl AudioProcessor for PassThrough {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone();
                true
            }
        }

        // Only three fan-in partitions, but 64 workers requested.
        let values: [f32; 3] = [1e20, 1.0, -1e20];
        fn build(values: &[f32; 3]) -> Graph {
            let mut g = Graph::new(llq::Queue::new().split().0);
            add_node(&mut g, 0, Box::new(PassThrough));
            add_node(&mut g, 1, Box::new(TestNode { tail_time: true }));
            for (i, &v) in values.iter().enumerate() {
                let src = 2 + i as u64;
                add_node(&mut g, src, Box::new(ConstSource { value: v }));
                add_edge(&mut g, src, 0);
            }
            g
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        let expected: Vec<u32> = build(&values)
            .render(&scope)
            .channel_data(0)
            .iter()
            .map(|x| x.to_bits())
            .collect();

        let mut g = build(&values);
        g.parallel_workers = 64;
        // Two quanta: reaching the assertion at all proves no deadlock.
        for q in 0..2 {
            let got: Vec<u32> = g
                .render_partitioned_threaded(&scope)
                .channel_data(0)
                .iter()
                .map(|x| x.to_bits())
                .collect();
            assert_eq!(
                got, expected,
                "64-worker threaded render (quantum {q}) must be byte-identical to serial"
            );
        }
    }

    // Breaking a feedback cycle clears the DelayNode writer's edge to its reader,
    // but the two halves share the delay line. The planner must keep them in one
    // partition (one thread), also when the feedback path runs through another
    // partition - which turns the writer/reader link into a partition-level
    // cycle that has to be collapsed.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_cycle_broken_delay_halves_share_a_partition() {
        use std::cell::Cell;

        #[derive(Debug, Clone)]
        struct ConstSource {
            value: f32,
        }
        impl AudioProcessor for ConstSource {
            fn process(
                &mut self,
                _inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s = self.value);
                true
            }
        }

        #[derive(Debug, Clone)]
        struct Scale {
            k: f32,
        }
        impl AudioProcessor for Scale {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                outputs[0] = inputs[0].clone();
                outputs[0]
                    .channel_data_mut(0)
                    .iter_mut()
                    .for_each(|s| *s *= self.k);
                true
            }
        }

        const WRITER: u64 = 3;
        const READER: u64 = 4;

        // dest(0) listener(1); src(2) -> writer(3) ~> reader(4) -> fb(5) -> writer,
        // reader -> dest, plus an unrelated source (6) so there is more than one
        // partition to spread. With `through_merge`, the feedback path detours
        // through a merge node (7, also fed by source 8) in its own partition.
        fn build(through_merge: bool) -> Graph {
            let mut g = Graph::new(llq::Queue::new().split().0);
            add_node(&mut g, 0, Box::new(Scale { k: 1.0 }));
            add_node(&mut g, 1, Box::new(TestNode { tail_time: true }));
            add_node(&mut g, 2, Box::new(ConstSource { value: 1.0 }));
            add_node(&mut g, WRITER, Box::new(Scale { k: 1.0 }));
            add_node(&mut g, READER, Box::new(Scale { k: 1.0 }));
            add_node(&mut g, 5, Box::new(Scale { k: 0.5 }));
            add_node(&mut g, 6, Box::new(ConstSource { value: 2.0 }));
            g.mark_cycle_breaker(AudioNodeId(WRITER));
            add_edge(&mut g, 2, WRITER);
            add_edge(&mut g, WRITER, READER);
            add_edge(&mut g, READER, 0);
            add_edge(&mut g, 6, 0);
            if through_merge {
                add_node(&mut g, 7, Box::new(Scale { k: 0.25 }));
                add_node(&mut g, 8, Box::new(ConstSource { value: 3.0 }));
                add_edge(&mut g, READER, 7);
                add_edge(&mut g, 8, 7);
                add_edge(&mut g, 7, 5);
            } else {
                add_edge(&mut g, READER, 5);
            }
            add_edge(&mut g, 5, WRITER);
            g
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        for through_merge in [false, true] {
            let serial_bits: Vec<Vec<u32>> = {
                let mut g = build(through_merge);
                (0..4)
                    .map(|_| {
                        g.render(&scope)
                            .channel_data(0)
                            .iter()
                            .map(|x| x.to_bits())
                            .collect()
                    })
                    .collect()
            };

            for workers in [2usize, 4] {
                let mut g = build(through_merge);
                g.parallel_workers = workers;
                for (q, expected) in serial_bits.iter().enumerate() {
                    let got: Vec<u32> = g
                        .render_partitioned_threaded(&scope)
                        .channel_data(0)
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                    assert_eq!(
                        &got, expected,
                        "through_merge={through_merge} workers={workers} quantum={q}: \
                         threaded render must be byte-identical to serial"
                    );
                }

                let plan = g.partition_plan.as_ref().expect("plan cached");
                assert!(
                    plan.well_formed,
                    "through_merge={through_merge}: partition DAG must stay acyclic"
                );
                let partition_of = |id: u64| {
                    plan.partitions
                        .iter()
                        .position(|p| p.contains(&AudioNodeId(id)))
                        .expect("node is planned")
                };
                assert_eq!(
                    partition_of(WRITER),
                    partition_of(READER),
                    "through_merge={through_merge}: delay writer and reader must share a partition"
                );
                assert!(
                    plan.partitions.len() > 1,
                    "co-locating the delay must not serialize the whole graph"
                );
            }
        }
    }

    // Once the feedback path of a cycle-broken delay is removed, no edge links
    // or orders the writer and reader any more (the cycle breaker's edges stay
    // cleared), so both land on the first DAG level. They still share the delay
    // line and must never run concurrently. The probes hold their "delay line"
    // for 2 ms so an overlap on two workers is all but certain to be observed.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_detached_cycle_breaker_halves_never_run_concurrently() {
        use std::cell::Cell;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        use std::thread::ThreadId;

        #[derive(Default)]
        struct DelayLine {
            busy: AtomicBool,
            overlapped: AtomicBool,
            threads: Mutex<Vec<(&'static str, ThreadId)>>,
        }

        struct Probe {
            name: &'static str,
            line: Arc<DelayLine>,
        }
        impl std::fmt::Debug for Probe {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Probe").field("name", &self.name).finish()
            }
        }
        impl AudioProcessor for Probe {
            fn process(
                &mut self,
                inputs: &[AudioRenderQuantum],
                outputs: &mut [AudioRenderQuantum],
                _params: AudioParamValues<'_>,
                _scope: &AudioWorkletGlobalScope,
            ) -> bool {
                if self.line.busy.swap(true, Ordering::SeqCst) {
                    self.line.overlapped.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
                self.line.busy.store(false, Ordering::SeqCst);
                self.line
                    .threads
                    .lock()
                    .unwrap()
                    .push((self.name, std::thread::current().id()));
                outputs[0] = inputs[0].clone();
                true
            }
        }

        let scope = AudioWorkletGlobalScope {
            current_frame: 0,
            current_time: 0.,
            sample_rate: 48000.,
            node_id: Cell::new(AudioNodeId(0)),
            event_sender: crossbeam_channel::unbounded().0,
        };

        let line = Arc::new(DelayLine::default());
        let mut g = Graph::new(llq::Queue::new().split().0);
        g.parallel_workers = 4;
        // dest(0, merge of 4 and 6) listener(1); src(2) -> writer(3) ~> reader(4)
        // -> fb(5) -> writer; reader -> dest; src(6) -> dest.
        add_node(&mut g, 0, Box::new(TestNode { tail_time: true }));
        add_node(&mut g, 1, Box::new(TestNode { tail_time: true }));
        add_node(&mut g, 2, Box::new(TestNode { tail_time: true }));
        let writer = Probe {
            name: "writer",
            line: Arc::clone(&line),
        };
        let reader = Probe {
            name: "reader",
            line: Arc::clone(&line),
        };
        add_node(&mut g, 3, Box::new(writer));
        add_node(&mut g, 4, Box::new(reader));
        add_node(&mut g, 5, Box::new(TestNode { tail_time: true }));
        add_node(&mut g, 6, Box::new(TestNode { tail_time: true }));
        g.mark_cycle_breaker(AudioNodeId(3));
        add_edge(&mut g, 2, 3);
        add_edge(&mut g, 3, 4);
        add_edge(&mut g, 4, 5);
        add_edge(&mut g, 5, 3);
        add_edge(&mut g, 4, 0);
        add_edge(&mut g, 6, 0);

        // Break the cycle (clears the writer -> reader edge), then detach the
        // feedback path so nothing orders the two halves any more.
        g.render_partitioned_threaded(&scope);
        g.remove_edge((AudioNodeId(5), 0), (AudioNodeId(3), 0));
        line.threads.lock().unwrap().clear();
        for _ in 0..8 {
            g.render_partitioned_threaded(&scope);
        }

        assert!(
            !line.overlapped.load(Ordering::SeqCst),
            "delay writer and reader ran concurrently on different workers"
        );
        let threads = line.threads.lock().unwrap();
        assert_eq!(threads.len(), 16, "both halves render every quantum");
        assert!(
            threads.iter().all(|&(_, t)| t == threads[0].1),
            "delay writer and reader must stay on one thread: {threads:?}"
        );
    }

    // The executor runs each worker's list in order and waits for producers by
    // their `done` marks, so the schedule itself must guarantee: every partition
    // exactly once, pinned partitions on worker 0, and no possible deadlock.
    // Checked on random DAGs by executing the lists with those waits.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_list_schedule_complete_pinned_and_deadlock_free() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n.max(1)
        };

        for case in 0..200 {
            let n = 1 + rand(60) as usize;
            // predecessors always have smaller indices (execution order)
            let preds: Vec<Vec<usize>> = (0..n)
                .map(|q| {
                    let mut ps: Vec<usize> = (0..rand(4))
                        .filter(|_| q > 0)
                        .map(|_| rand(q as u64) as usize)
                        .collect();
                    ps.sort_unstable();
                    ps.dedup();
                    ps
                })
                .collect();
            let costs: Vec<u64> = (0..n).map(|_| 1 + rand(10_000)).collect();
            let pinned: Vec<bool> = (0..n).map(|_| rand(10) == 0).collect();
            let workers = 1 + rand(8) as usize;

            let (partition_worker, lists) = list_schedule(&costs, &preds, &pinned, workers);

            assert_eq!(lists.len(), workers, "case {case}");
            let mut seen = vec![false; n];
            for (w, list) in lists.iter().enumerate() {
                for &q in list {
                    assert!(!seen[q], "case {case}: partition {q} scheduled twice");
                    seen[q] = true;
                    assert_eq!(partition_worker[q], w, "case {case}");
                    if pinned[q] {
                        assert_eq!(w, 0, "case {case}: pinned partition {q} off worker 0");
                    }
                }
            }
            assert!(
                seen.iter().all(|&s| s),
                "case {case}: partition not scheduled"
            );

            // Execute the lists: each worker may only run its next partition once
            // all its producers are done. Must finish without getting stuck.
            let mut done = vec![false; n];
            let mut next = vec![0usize; workers];
            let mut remaining = n;
            while remaining > 0 {
                let mut progressed = false;
                for w in 0..workers {
                    if let Some(&q) = lists[w].get(next[w]) {
                        if preds[q].iter().all(|&p| done[p]) {
                            done[q] = true;
                            next[w] += 1;
                            remaining -= 1;
                            progressed = true;
                        }
                    }
                }
                assert!(progressed, "case {case}: schedule deadlocks");
            }
        }
    }
}
