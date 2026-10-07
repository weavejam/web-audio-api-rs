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

impl Graph {
    pub fn new(reclaim_id_channel: llq::Producer<AudioNodeId>) -> Self {
        // Opt-in to the partitioned render path (native only). Read once at graph
        // construction so the hot render loop never touches the environment.
        #[cfg(not(target_arch = "wasm32"))]
        let partitioned =
            std::env::var("WEB_AUDIO_RS_PARTITIONED").is_ok_and(|v| v == "1" || v == "true");
        #[cfg(target_arch = "wasm32")]
        let partitioned = false;

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
        }
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
            ordered: self.ordered.iter().map(|id| id.0).collect(),
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
        self.partition_plan = None; // topology changed, recompute partitions
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
        self.partition_plan = None;
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
    }
    pub fn set_channel_count_mode(&mut self, index: AudioNodeId, v: ChannelCountMode) {
        self.nodes
            .get_unchecked_mut(index)
            .channel_config
            .count_mode = v;
    }
    pub fn set_channel_interpretation(&mut self, index: AudioNodeId, v: ChannelInterpretation) {
        self.nodes
            .get_unchecked_mut(index)
            .channel_config
            .interpretation = v;
    }

    pub fn route_message(&mut self, index: AudioNodeId, msg: &mut dyn Any) {
        self.nodes.get_unchecked_mut(index).processor.onmessage(msg);
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
                // clear the outgoing edges of the nodes that have been recognized as cycle breaker
                cycle_breakers.iter().for_each(|node_id| {
                    self.nodes
                        .get_unchecked_mut(*node_id)
                        .outgoing_edges
                        .clear();
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
    }

    /// Render a single audio quantum, dispatching to the partitioned (multicore)
    /// path when enabled (see [`Graph::partitioned`]) or the serial path otherwise.
    /// Both produce byte-identical output. This is the entry point the render
    /// thread should call.
    pub fn render_quantum(&mut self, scope: &AudioWorkletGlobalScope) -> &AudioRenderQuantum {
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
                let catch_me = AssertUnwindSafe(|| node.process(params, scope));

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
                    node.get_mut()
                        .outgoing_edges
                        .retain(|e| e.other_id != *index);
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
        let is_merge: Vec<bool> = audio_in_degree.iter().map(|&d| d >= 2).collect();

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
                    // Consumer is not in `ordered` (e.g. excluded as part of a cycle).
                    // The serial render pushes into such a node's input buffer but never
                    // processes it, so it cannot affect the destination output. Skip it:
                    // it must not create a boundary slot or a partition-DAG edge.
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

        // Reorder partitions into execution order (if well-formed).
        let ordered_partitions: Vec<Vec<AudioNodeId>> = if well_formed {
            order
                .iter()
                .map(|&g| std::mem::take(&mut partitions[g]))
                .collect()
        } else {
            Vec::new()
        };

        PartitionPlan {
            partitions: ordered_partitions,
            is_merge,
            copy_out,
            merge_lookup,
            boundary_count,
            well_formed,
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
                    let catch_me = AssertUnwindSafe(|| node.process(params, scope));
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
                    self.reclaim_id_channel
                        .push(node.reclaim_id.take().unwrap());
                    node.processor.before_drop(scope);
                    drop(node);

                    nodes_dropped = true;

                    self.nodes.values_mut().for_each(|node| {
                        node.get_mut()
                            .outgoing_edges
                            .retain(|e| e.other_id != index);
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
            // leave self.partition_plan == None -> recompute next quantum
        } else {
            self.partition_plan = Some(plan);
        }

        // Return the output buffer of destination node
        &self.nodes.get_unchecked_mut(AudioNodeId(0)).outputs[0]
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
}
