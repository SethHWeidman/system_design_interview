# dynamo_kv.py - A simplified Dynamo-style key-value store for learning
# (Saved from the notebook execution. If you want to run the live demo,
# run this file: python dynamo_kv.py)

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Iterable, Set
import hashlib, bisect, time

@dataclass(frozen=True)
class VectorClock:
    clock: Dict[str, int] = field(default_factory=dict)
    def increment(self, node_id: str) -> "VectorClock":
        c = dict(self.clock); c[node_id] = c.get(node_id, 0) + 1; return VectorClock(c)
    def merge(self, other: "VectorClock") -> "VectorClock":
        keys = set(self.clock) | set(other.clock)
        return VectorClock({k: max(self.clock.get(k, 0), other.clock.get(k, 0)) for k in keys})
    def compare(self, other: "VectorClock") -> str:
        a, b = self.clock, other.clock; keys = set(a) | set(b)
        a_le_b = True; b_le_a = True; strictly_less = False; strictly_greater = False
        for k in keys:
            av = a.get(k, 0); bv = b.get(k, 0)
            if av > bv: a_le_b = False; strictly_greater = True
            elif av < bv: b_le_a = False; strictly_less = True
        if a == b: return "equal"
        if a_le_b and strictly_less: return "descends"
        if b_le_a and strictly_greater: return "dominates"
        return "concurrent"
    def __str__(self): return str(self.clock)

@dataclass
class VersionedValue:
    value: Any; vclock: VectorClock; timestamp: float = field(default_factory=lambda: time.time())
    def conflicts_with(self, other: "VersionedValue") -> bool:
        return self.vclock.compare(other.vclock) == "concurrent"

class ConsistentHashRing:
    def __init__(self, virtual_nodes_per_node: int = 32):
        self.virtual_nodes_per_node = virtual_nodes_per_node
        self.ring: List[Tuple[int, str]] = []; self.nodes: Set[str] = set()
    @staticmethod
    def _hash_key(key: str) -> int:
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    def add_node(self, node_id: str):
        if node_id in self.nodes: return
        self.nodes.add(node_id)
        for i in range(self.virtual_nodes_per_node):
            h = self._hash_key(f'{node_id}#{i}'); bisect.insort(self.ring, (h, node_id))
    def remove_node(self, node_id: str):
        if node_id not in self.nodes: return
        self.nodes.remove(node_id); self.ring = [(h, n) for (h, n) in self.ring if n != node_id]
    def get_preference_list(self, key: str, n: int) -> List[str]:
        if not self.ring: return []
        h = self._hash_key(key); idx = bisect.bisect(self.ring, (h, chr(255)))
        result, seen = [], set(); i = 0
        while len(result) < n and i < len(self.ring) * 2:
            p = (idx + i) % len(self.ring); node_id = self.ring[p][1]
            if node_id not in seen: result.append(node_id); seen.add(node_id)
            i += 1
        return result

class Node:
    def __init__(self, node_id: str):
        self.node_id = node_id; self.alive = True
        self.store: Dict[str, List[VersionedValue]] = {}
        self.hints: Dict[str, List[Tuple[str, VersionedValue]]] = {}
    def is_alive(self) -> bool: return self.alive
    def set_alive(self, alive: bool): self.alive = alive
    def apply_put(self, key: str, vv: VersionedValue):
        siblings = self.store.get(key, []); new_siblings: List[VersionedValue] = []; dominated_by_existing = False
        for s in siblings:
            rel = vv.vclock.compare(s.vclock)
            if rel == 'dominates': continue
            elif rel == 'descends': dominated_by_existing = True; new_siblings.append(s)
            elif rel == 'equal':
                if s.timestamp >= vv.timestamp: new_siblings.append(s)
                else: new_siblings.append(vv)
            else: new_siblings.append(s)
        if not dominated_by_existing:
            equal_present = any(vv.vclock.compare(s.vclock) == 'equal' for s in new_siblings)
            if not equal_present: new_siblings.append(vv)
        self.store[key] = new_siblings
    def get_versions(self, key: str) -> List[VersionedValue]:
        return list(self.store.get(key, []))
    def add_hint(self, target_node_id: str, key: str, vv: VersionedValue):
        self.hints.setdefault(target_node_id, []).append((key, vv))
    def drain_hints_for(self, target_node_id: str) -> List[Tuple[str, VersionedValue]]:
        return self.hints.pop(target_node_id, [])

class Cluster:
    def __init__(self, node_ids: Iterable[str], N: int = 3, R: int = 2, W: int = 2, virtual_nodes: int = 32):
        assert N >= 1 and R >= 1 and W >= 1
        self.N, self.R, self.W = N, R, W
        self.nodes: Dict[str, Node] = {nid: Node(nid) for nid in node_ids}
        self.ring = ConsistentHashRing(virtual_nodes_per_node=virtual_nodes)
        for nid in node_ids: self.ring.add_node(nid)
    def node(self, node_id: str) -> Node: return self.nodes[node_id]
    def alive_nodes(self) -> List[str]: return [nid for nid, n in self.nodes.items() if n.is_alive()]
    def preference_list(self, key: str) -> List[str]: return self.ring.get_preference_list(key, self.N)
    def _sloppy_targets(self, key: str) -> List[Tuple[str, Optional[str]]]:
        preferred = self.preference_list(key); storage_targets: List[Tuple[str, Optional[str]]] = []; used: Set[str] = set()
        for p in preferred:
            if self.node(p).is_alive(): storage_targets.append((p, None)); used.add(p)
        if len(storage_targets) < self.N:
            i = 0
            while len(storage_targets) < self.N and i < len(self.ring.ring) * 2:
                cand = self.ring.ring[i % len(self.ring.ring)][1]
                if cand not in used and self.node(cand).is_alive():
                    down_originals = [p for p in preferred if not self.node(p).is_alive() and p not in [orig for (_, orig) in storage_targets if orig]]
                    orig = down_originals[0] if down_originals else None
                    storage_targets.append((cand, orig)); used.add(cand)
                i += 1
        return storage_targets[:self.N]
    def put(self, key: str, value: Any, context: Optional[VectorClock] = None, coordinator_id: Optional[str] = None) -> Dict[str, Any]:
        coord = coordinator_id or (self.preference_list(key)[0] if self.preference_list(key) else None)
        if coord not in self.nodes or not self.node(coord).is_alive():
            alive = self.alive_nodes()
            if not alive: raise RuntimeError('No alive coordinator available')
            coord = alive[0]
        base = context or VectorClock(); vv = VersionedValue(value=value, vclock=base.increment(coord))
        targets = self._sloppy_targets(key); acks = 0; written_to: List[Tuple[str, Optional[str]]] = []
        for storage_nid, original in targets:
            n = self.node(storage_nid)
            if not n.is_alive(): continue
            n.apply_put(key, vv)
            if original: n.add_hint(original, key, vv)
            acks += 1; written_to.append((storage_nid, original))
        success = acks >= self.W
        return {'key': key, 'value': value, 'vector_clock': vv.vclock.clock, 'coordinator': coord,
                'preferred_replicas': self.preference_list(key), 'actual_writes': written_to,
                'acks': acks, 'success': success,
                'note': 'Sloppy quorum used' if any(orig for (_, orig) in written_to) else 'Strict quorum'}
    def get(self, key: str, coordinator_id: Optional[str] = None) -> Dict[str, Any]:
        coord = coordinator_id or (self.preference_list(key)[0] if self.preference_list(key) else None)
        if coord not in self.nodes or not self.node(coord).is_alive():
            alive = self.alive_nodes()
            if not alive: raise RuntimeError('No alive coordinator available')
            coord = alive[0]
        preferred = self.preference_list(key)
        responses: List[Tuple[str, List[VersionedValue]]] = []; contacted: List[str] = []
        for nid in preferred:
            if self.node(nid).is_alive():
                versions = self.node(nid).get_versions(key)
                responses.append((nid, versions)); contacted.append(nid)
                if len(responses) >= self.R: break
        all_versions = []; [all_versions.extend(vs) for _, vs in responses]
        reduced: List[VersionedValue] = []
        for vv in all_versions:
            dominated = False; to_keep = []
            for kept in reduced:
                rel = vv.vclock.compare(kept.vclock)
                if rel == 'dominates': continue
                elif rel == 'descends': dominated = True; to_keep.append(kept)
                elif rel == 'equal':
                    if kept.timestamp >= vv.timestamp: to_keep.append(kept)
                else: to_keep.append(kept)
            if not dominated:
                reduced = [k for k in to_keep if k.vclock.compare(vv.vclock) != 'descends']
                if not any(vv.vclock.compare(k.vclock) == 'equal' for k in reduced): reduced.append(vv)
        for nid in contacted:
            node_versions = self.node(nid).get_versions(key)
            for rv in reduced:
                if not any(rv.vclock.compare(v.vclock) in ('equal', 'descends') for v in node_versions):
                    self.node(nid).apply_put(key, rv)
        return {'key': key, 'values': [v.value for v in reduced], 'siblings': len(reduced) > 1,
                'vector_clocks': [v.vclock.clock for v in reduced], 'coordinator': coord,
                'contacted': contacted, 'preferred_replicas': preferred}
    def fail_node(self, node_id: str): self.node(node_id).set_alive(False)
    def recover_node(self, node_id: str): self.node(node_id).set_alive(True)
    def drain_hinted_handoffs(self, target_node_id: str) -> Dict[str, Any]:
        delivered = []
        for nid, node in self.nodes.items():
            if not node.is_alive(): continue
            hints = node.drain_hints_for(target_node_id)
            for key, vv in hints:
                if self.node(target_node_id).is_alive():
                    self.node(target_node_id).apply_put(key, vv); delivered.append((nid, key, vv.value))
                else:
                    node.add_hint(target_node_id, key, vv)
        return {'delivered': delivered}
    def dump_key_locations(self, key: str) -> List[Tuple[str, List[Any]]]:
        out = []
        for nid in self.nodes:
            vals = [vv.value for vv in self.node(nid).get_versions(key)]
            if vals: out.append((nid, vals))
        return sorted(out, key=lambda x: x[0])
    def show_status(self) -> str:
        parts = []
        for nid, node in sorted(self.nodes.items()):
            parts.append(f"{nid}: {'UP' if node.is_alive() else 'DOWN'} (keys={len(node.store)})")
        return ' | '.join(parts)

def _demo():
    cluster = Cluster(['A','B','C','D'], N=3, R=2, W=2, virtual_nodes=16)
    print('Cluster status:', cluster.show_status())
    key = 'user:1'
    print('\nPUT user:1 = Alice')
    print(cluster.put(key, 'Alice'))
    print('Locations:', cluster.dump_key_locations(key))
    print('\nGET user:1')
    print(cluster.get(key))
    failed = cluster.preference_list(key)[1]; print('\nFail', failed); cluster.fail_node(failed)
    print('Cluster status:', cluster.show_status())
    print('\nPUT user:1 = Alice v2 (during failure)')
    print(cluster.put(key, 'Alice v2'))
    print('Locations:', cluster.dump_key_locations(key))
    print('\nRecover + drain hints'); cluster.recover_node(failed)
    print(cluster.drain_hinted_handoffs(failed))
    print('Locations:', cluster.dump_key_locations(key))
    print('\nGET user:1 (read-repair)'); print(cluster.get(key))
    key2 = 'cart:42'; print('\nConflict demo on', key2)
    print(cluster.put(key2, {'items':['base']}, coordinator_id='A'))
    r = cluster.get(key2); base_vc = VectorClock(r['vector_clocks'][0])
    print(cluster.put(key2, {'items':['base','A']}, context=base_vc, coordinator_id='A'))
    print(cluster.put(key2, {'items':['base','B']}, context=base_vc, coordinator_id='B'))
    r2 = cluster.get(key2); print('GET ->', r2)
    # resolve
    items = []; [items.extend(v['items']) for v in r2['values']]
    items = list(dict.fromkeys(items))
    vc = VectorClock()
    for vc_d in r2['vector_clocks']: vc = vc.merge(VectorClock(vc_d))
    print('Resolved PUT:', cluster.put(key2, {'items': items}, context=vc, coordinator_id='C'))
    print('Final GET:', cluster.get(key2))

if __name__ == '__main__':
    _demo()
