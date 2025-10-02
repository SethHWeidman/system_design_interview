"""Teaching-friendly Dynamo-style key-value store demonstration.

This module mirrors the behaviour of the Dynamo-inspired key-value store
implemented in ``dynamo_kv.py`` but annotates each moving part with
explanatory comments. It focuses on explaining common Dynamo jargon such as
vector clocks, consistent hashing, hinted handoff, sloppy quorums, and
read-repair, making the control flow easier to follow for educational use.
"""

import bisect
import dataclasses
import hashlib
import pprint
import time
import typing


@dataclasses.dataclass(frozen=True)
class VectorClock:
    """Causal version counter keyed by node identifier.

    Each node keeps its own counter. By comparing these per-node counters we can
    tell whether one version causally follows another ("dominates"), is older
    ("descends"), or is in conflict ("concurrent").
    """

    clock: dict[str, int] = dataclasses.field(default_factory=dict)

    def increment(self, node_id: str) -> "VectorClock":
        """Return a new vector clock with ``node_id``'s counter bumped by one."""

        c = dict(self.clock)
        c[node_id] = c.get(node_id, 0) + 1
        return VectorClock(c)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Combine clocks by taking the maximum counter per node."""

        keys = set(self.clock) | set(other.clock)
        return VectorClock(
            {k: max(self.clock.get(k, 0), other.clock.get(k, 0)) for k in keys}
        )

    def compare(self, other: "VectorClock") -> str:
        """Describe ordering relative to ``other``.

        Returns one of:
        * ``"equal"``        – same counters everywhere.
        * ``"dominates"``    – this clock causally follows ``other``.
        * ``"descends"``     – this clock is older than ``other``.
        * ``"concurrent"``   – neither version knows about the other; conflict.
        """

        a, b = self.clock, other.clock
        keys = set(a) | set(b)
        a_le_b = True
        b_le_a = True
        strictly_less = False
        strictly_greater = False
        for k in keys:
            av = a.get(k, 0)
            bv = b.get(k, 0)
            if av > bv:
                a_le_b = False
                strictly_greater = True
            elif av < bv:
                b_le_a = False
                strictly_less = True
        if a == b:
            return "equal"
        if a_le_b and strictly_less:
            return "descends"
        if b_le_a and strictly_greater:
            return "dominates"
        return "concurrent"

    def __str__(self) -> str:
        return str(self.clock)


@dataclasses.dataclass
class VersionedValue:
    """A stored value paired with its vector clock and arrival timestamp."""

    value: typing.Any
    vclock: VectorClock
    timestamp: float = dataclasses.field(default_factory=lambda: time.time())

    def conflicts_with(self, other: "VersionedValue") -> bool:
        """Return True if neither version is aware of the other (concurrency)."""

        return self.vclock.compare(other.vclock) == "concurrent"


class ConsistentHashRing:
    """Minimal consistent hashing implementation with virtual nodes.

    *Consistent hashing* spreads keys across nodes so that churn only remaps a
    small portion of keys. *Virtual nodes* (a node mapped to many points on the
    ring) smooth out distribution hotspots.
    """

    def __init__(self, virtual_nodes_per_node: int = 32):
        self.virtual_nodes_per_node = virtual_nodes_per_node
        self.ring: list[tuple[int, str]] = []  # Sorted list of (hash, node_id)
        self.nodes: set[str] = set()

    @staticmethod
    def _hash_key(key: str) -> int:
        """Hash helper that returns a large integer suitable for ring placement."""

        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_node(self, node_id: str):
        """Place ``node_id`` on the ring at multiple points for load balancing."""

        if node_id in self.nodes:
            return
        self.nodes.add(node_id)
        for i in range(self.virtual_nodes_per_node):
            h = self._hash_key(f"{node_id}#{i}")
            bisect.insort(self.ring, (h, node_id))  # Keep ring sorted

    def remove_node(self, node_id: str):
        """Remove all virtual nodes associated with ``node_id``."""

        if node_id not in self.nodes:
            return
        self.nodes.remove(node_id)
        self.ring = [(h, n) for (h, n) in self.ring if n != node_id]

    def get_preference_list(self, key: str, n: int) -> list[str]:
        """Return the top ``n`` replicas responsible for ``key``.

        This is Dynamo's *preference list* that supplies the ideal replica set
        (before accounting for failures).
        """

        if not self.ring:
            return []
        h = self._hash_key(key)
        idx = bisect.bisect(self.ring, (h, chr(255)))
        result, seen = [], set()
        i = 0
        while len(result) < n and i < len(self.ring) * 2:
            p = (idx + i) % len(self.ring)
            node_id = self.ring[p][1]
            if node_id not in seen:
                result.append(node_id)
                seen.add(node_id)
            i += 1
        return result


class Node:
    """A single Dynamo storage node with local replicas and hinted handoff slots."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.alive = True
        self.store: dict[str, list[VersionedValue]] = {}
        # Hinted handoff buffer: key -> replicas we are temporarily storing for.
        self.hints: dict[str, list[tuple[str, VersionedValue]]] = {}

    def is_alive(self) -> bool:
        return self.alive

    def set_alive(self, alive: bool):
        self.alive = alive

    def apply_put(self, key: str, vv: VersionedValue):
        """Merge ``vv`` into the local replica, keeping only relevant siblings."""

        siblings = self.store.get(key, [])
        new_siblings: list[VersionedValue] = []
        dominated_by_existing = False
        for s in siblings:
            rel = vv.vclock.compare(s.vclock)
            if rel == "dominates":
                # New write supersedes the sibling, so drop the older copy.
                continue
            if rel == "descends":
                dominated_by_existing = True
                new_siblings.append(s)
            elif rel == "equal":
                # Keep whichever version arrived last to avoid duplicate siblings.
                if s.timestamp >= vv.timestamp:
                    new_siblings.append(s)
                else:
                    new_siblings.append(vv)
            else:  # concurrent
                new_siblings.append(s)
        if not dominated_by_existing:
            equal_present = any(
                vv.vclock.compare(s.vclock) == "equal" for s in new_siblings
            )
            if not equal_present:
                new_siblings.append(vv)
        self.store[key] = new_siblings

    def get_versions(self, key: str) -> list[VersionedValue]:
        """Return a shallow copy so callers cannot mutate internal state."""

        return list(self.store.get(key, []))

    def add_hint(self, target_node_id: str, key: str, vv: VersionedValue):
        """Record data we are holding on behalf of ``target_node_id``.

        This is Dynamo's *hinted handoff*: a healthy node temporarily stores
        replicas for an unavailable peer, delivering them once it recovers.
        """

        self.hints.setdefault(target_node_id, []).append((key, vv))

    def drain_hints_for(self, target_node_id: str) -> list[tuple[str, VersionedValue]]:
        """Return and remove hints destined for ``target_node_id``."""

        return self.hints.pop(target_node_id, [])


class Cluster:
    """High-level Dynamo cluster simulacrum supporting N/R/W quorum semantics.

    Parameters
    ----------
    N : replication factor (number of replicas stored per key).
    R : number of successful replicas required for a read quorum.
    W : number of successful replicas required for a write quorum.

    The cluster also models *sloppy quorums* (writing to healthy fallbacks when
    a preferred replica is down) and *read repair* (background reconciliation
    performed during reads).
    """

    def __init__(
        self,
        node_ids: typing.Iterable[str],
        N: int = 3,
        R: int = 2,
        W: int = 2,
        virtual_nodes: int = 32,
    ):
        assert N >= 1 and R >= 1 and W >= 1
        self.N, self.R, self.W = N, R, W
        self.nodes: dict[str, Node] = {nid: Node(nid) for nid in node_ids}
        self.ring = ConsistentHashRing(virtual_nodes_per_node=virtual_nodes)
        for nid in node_ids:
            self.ring.add_node(nid)

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def alive_nodes(self) -> list[str]:
        return [nid for nid, n in self.nodes.items() if n.is_alive()]

    def preference_list(self, key: str) -> list[str]:
        return self.ring.get_preference_list(key, self.N)

    def _sloppy_targets(self, key: str) -> list[tuple[str, typing.Optional[str]]]:
        """Pick replicas for ``key``, falling back to healthy nodes if needed."""

        preferred = self.preference_list(key)
        storage_targets: list[tuple[str, typing.Optional[str]]] = []
        used: set[str] = set()
        for p in preferred:
            if self.node(p).is_alive():
                storage_targets.append((p, None))
                used.add(p)
        if len(storage_targets) < self.N:
            i = 0
            while len(storage_targets) < self.N and i < len(self.ring.ring) * 2:
                cand = self.ring.ring[i % len(self.ring.ring)][1]
                if cand not in used and self.node(cand).is_alive():
                    down_originals = [
                        p
                        for p in preferred
                        if not self.node(p).is_alive()
                        and p not in [orig for (_, orig) in storage_targets if orig]
                    ]
                    orig = down_originals[0] if down_originals else None
                    storage_targets.append((cand, orig))
                    used.add(cand)
                i += 1
        return storage_targets[: self.N]

    def put(
        self,
        key: str,
        value: typing.Any,
        context: typing.Optional[VectorClock] = None,
        coordinator_id: typing.Optional[str] = None,
    ) -> dict[str, typing.Any]:
        """Store ``value`` under ``key`` honouring the ``W`` write quorum."""

        coord = coordinator_id or (
            self.preference_list(key)[0] if self.preference_list(key) else None
        )
        if coord not in self.nodes or not self.node(coord).is_alive():
            alive = self.alive_nodes()
            if not alive:
                raise RuntimeError("No alive coordinator available")
            coord = alive[0]
        base = context or VectorClock()
        vv = VersionedValue(value=value, vclock=base.increment(coord))
        targets = self._sloppy_targets(key)
        acks = 0
        written_to: list[tuple[str, typing.Optional[str]]] = []
        for storage_nid, original in targets:
            n = self.node(storage_nid)
            if not n.is_alive():
                continue
            n.apply_put(key, vv)
            if original:
                n.add_hint(original, key, vv)
            acks += 1
            written_to.append((storage_nid, original))
        success = acks >= self.W
        return {
            "key": key,
            "value": value,
            "vector_clock": vv.vclock.clock,
            "coordinator": coord,
            "preferred_replicas": self.preference_list(key),
            "actual_writes": written_to,
            "acks": acks,
            "success": success,
            "note": (
                "Sloppy quorum used"
                if any(orig for (_, orig) in written_to)
                else "Strict quorum"
            ),
        }

    def get(
        self, key: str, coordinator_id: typing.Optional[str] = None
    ) -> dict[str, typing.Any]:
        """Fetch ``key`` and perform read repair if replicas disagree."""

        coord = coordinator_id or (
            self.preference_list(key)[0] if self.preference_list(key) else None
        )
        if coord not in self.nodes or not self.node(coord).is_alive():
            alive = self.alive_nodes()
            if not alive:
                raise RuntimeError("No alive coordinator available")
            coord = alive[0]
        preferred = self.preference_list(key)
        responses: list[tuple[str, list[VersionedValue]]] = []
        contacted: list[str] = []
        for nid in preferred:
            if self.node(nid).is_alive():
                versions = self.node(nid).get_versions(key)
                responses.append((nid, versions))
                contacted.append(nid)
                if len(responses) >= self.R:
                    break
        all_versions = []
        for _, vs in responses:
            all_versions.extend(vs)
        reduced: list[VersionedValue] = []
        for vv in all_versions:
            dominated = False
            to_keep = []
            for kept in reduced:
                rel = vv.vclock.compare(kept.vclock)
                if rel == "dominates":
                    continue
                if rel == "descends":
                    dominated = True
                    to_keep.append(kept)
                elif rel == "equal":
                    if kept.timestamp >= vv.timestamp:
                        to_keep.append(kept)
                else:
                    to_keep.append(kept)
            if not dominated:
                reduced = [
                    k for k in to_keep if k.vclock.compare(vv.vclock) != "descends"
                ]
                if not any(vv.vclock.compare(k.vclock) == "equal" for k in reduced):
                    reduced.append(vv)
        for nid in contacted:
            node_versions = self.node(nid).get_versions(key)
            for rv in reduced:
                if not any(
                    rv.vclock.compare(v.vclock) in ("equal", "descends")
                    for v in node_versions
                ):
                    self.node(nid).apply_put(key, rv)
        return {
            "key": key,
            "values": [v.value for v in reduced],
            "siblings": len(reduced) > 1,
            "vector_clocks": [v.vclock.clock for v in reduced],
            "coordinator": coord,
            "contacted": contacted,
            "preferred_replicas": preferred,
        }

    def fail_node(self, node_id: str):
        """Simulate a node failure (used in demonstrations)."""

        self.node(node_id).set_alive(False)

    def recover_node(self, node_id: str):
        """Bring a failed node back online."""

        self.node(node_id).set_alive(True)

    def drain_hinted_handoffs(self, target_node_id: str) -> dict[str, typing.Any]:
        """Attempt delivery of outstanding hints to ``target_node_id``."""

        delivered = []
        for nid, node in self.nodes.items():
            if not node.is_alive():
                continue
            hints = node.drain_hints_for(target_node_id)
            for key, vv in hints:
                if self.node(target_node_id).is_alive():
                    self.node(target_node_id).apply_put(key, vv)
                    delivered.append((nid, key, vv.value))
                else:
                    # If the target went down again mid-transfer, keep holding it.
                    node.add_hint(target_node_id, key, vv)
        return {"delivered": delivered}

    def dump_key_locations(self, key: str) -> list[tuple[str, list[typing.Any]]]:
        """Show which nodes currently store replicas of ``key``."""

        out = []
        for nid in self.nodes:
            vals = [vv.value for vv in self.node(nid).get_versions(key)]
            if vals:
                out.append((nid, vals))
        return sorted(out, key=lambda x: x[0])

    def show_status(self) -> str:
        """Human-friendly status string describing node health and load."""

        parts = []
        for nid, node in sorted(self.nodes.items()):
            parts.append(
                f"{nid}: {'UP' if node.is_alive() else 'DOWN'} (keys={len(node.store)})"
            )
        return " | ".join(parts)


def _demo():
    """Step-by-step walkthrough of the core Dynamo design ideas."""

    cluster = Cluster(["A", "B", "C", "D"], N=3, R=2, W=2, virtual_nodes=16)
    print("=== Initial cluster state ===")
    print("Replicas and health before any requests:", cluster.show_status())

    key = "user:1"
    print(
        f"\n--- Step 1: Initial write for {key} ---"
        "We store 'Alice' and expect a strict quorum because all replicas are up."
    )
    pprint.pprint(cluster.put(key, "Alice"))
    print("Replica contents after write:")
    pprint.pprint(cluster.dump_key_locations(key))

    print(
        f"\n--- Step 2: Read {key} with all replicas healthy ---"
        "A read quorum (R=2) should succeed without conflicts."
    )
    pprint.pprint(cluster.get(key))

    failed = cluster.preference_list(key)[1]
    print(f"\n--- Step 3: Simulate failure of replica {failed} ---")
    cluster.fail_node(failed)
    print("Cluster health after failure:", cluster.show_status())

    print("\n--- Step 4: Write during failure (sloppy quorum & hinted handoff) ---")
    print(
        "We write 'Alice v2'; the coordinator should fall back to healthy nodes and "
        "leave a hint for the down replica."
    )
    pprint.pprint(cluster.put(key, "Alice v2"))
    print("Replica contents after sloppy quorum write:")
    pprint.pprint(cluster.dump_key_locations(key))

    print("\n--- Step 5: Recover failed replica and drain hints ---")
    cluster.recover_node(failed)
    print("Cluster health after recovery:", cluster.show_status())
    print("Deliver stored hints back to the original replica:")
    pprint.pprint(cluster.drain_hinted_handoffs(failed))
    print("Replica contents after hinted handoff:")
    pprint.pprint(cluster.dump_key_locations(key))

    print("\n--- Step 6: Read after repair to demonstrate read-repair ---")
    print("The read should now see a single value and repair any stale replicas.")
    pprint.pprint(cluster.get(key))

    key2 = "cart:42"
    print(
        f"\n--- Step 7: Demonstrate conflicting writes on {key2} ---"
        "We start with a clean cart, then perform concurrent updates from nodes A and "
        "B using the same vector clock context to force a conflict."
    )
    pprint.pprint(cluster.put(key2, {"items": ["base"]}, coordinator_id="A"))

    r = cluster.get(key2)
    base_vc = VectorClock(r["vector_clocks"][0])
    pprint.pprint(
        cluster.put(key2, {"items": ["base", "A"]}, context=base_vc, coordinator_id="A")
    )
    pprint.pprint(
        cluster.put(key2, {"items": ["base", "B"]}, context=base_vc, coordinator_id="B")
    )

    print("\nRead back the cart to inspect divergent siblings:")
    r2 = cluster.get(key2)
    pprint.pprint(r2)

    print("\n--- Step 8: Client-side resolution and merged write ---")
    # CONFLICT RESOLUTION STRATEGY: Union (Set Merge)
    #
    # When the GET returns siblings (concurrent conflicting values), the client must
    # choose a resolution strategy. This example uses a "union" approach: we merge all
    # items from all siblings into a single list.
    #
    # For a shopping cart, this makes sense: if two concurrent operations added different
    # items, we want to preserve both additions.
    #
    # ALTERNATIVE STRATEGIES:
    # The resolution strategy is application-specific and could be different:
    # - Intersection: Keep only items present in ALL siblings
    #   items = list(set.intersection(*[set(v["items"]) for v in r2["values"]]))
    # - Last-write-wins: Choose the value with the latest timestamp
    # - Custom business logic: Apply domain-specific merge rules
    #
    # The key point: Dynamo doesn't dictate HOW to resolve conflicts, only that the
    # client MUST resolve them before writing back to prevent unbounded sibling growth.
    items = []
    for v in r2["values"]:
        items.extend(v["items"])
    items = list(dict.fromkeys(items))  # Preserve original order while deduping

    # Merge all vector clocks from the conflicting siblings to create a context that
    # causally dominates both branches. This tells the cluster: "I've seen both
    # conflicting versions and am now reconciling them."
    vc = VectorClock()
    for vc_d in r2["vector_clocks"]:
        vc = vc.merge(VectorClock(vc_d))

    # Write the reconciled value back with the merged vector clock context. This new
    # write will supersede both conflicting siblings.
    pprint.pprint(cluster.put(key2, {"items": items}, context=vc, coordinator_id="C"))
    print("Final GET after reconciliation:")
    pprint.pprint(cluster.get(key2))


if __name__ == "__main__":
    _demo()
