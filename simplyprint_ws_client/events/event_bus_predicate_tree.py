from typing import NamedTuple, List, Dict, Set, Iterator, TypeVar, Generic, Iterable

from simplyprint_ws_client.events.predicate import Predicate, Constant


class PredicateTreeNode(NamedTuple):
    predicate: Predicate
    predicates: List["PredicateTreeNode"]
    resources: Set[int]

    def contains(self, item: Predicate):
        if not isinstance(item, Predicate):
            return False

        for entry in self.predicates:
            if entry.predicate == item:
                return True

        return False

    def push(self, resource_id: int, *predicates: Predicate):
        if len(predicates) == 0:
            self.resources.add(resource_id)
            return

        predicates = list(predicates)
        predicate = predicates.pop(0)

        for entry in self.predicates:
            if entry.predicate == predicate:
                entry.push(resource_id, *predicates)
                return

        entry = PredicateTreeNode(predicate, [], set())
        self.predicates.append(entry)
        entry.push(resource_id, *predicates)


_TResource = TypeVar("_TResource")


class EventBusPredicateTree(Generic[_TResource]):
    """Store data based on predicate trees. Useful for very dynamic event systems."""

    root: PredicateTreeNode
    resources: Dict[int, _TResource]
    current_resource_id: int = 0

    def __init__(self):
        # Root predicate node, we skip this and access the children directly.
        self.root = PredicateTreeNode(Constant(True), [], set())
        self.resources = {}

    def add(self, resource: _TResource, *predicates) -> int:
        self.current_resource_id = next_resource_id = self.current_resource_id + 1
        self.resources[next_resource_id] = resource
        self.root.push(next_resource_id, *predicates)
        return next_resource_id

    def remove(self, resource: _TResource):
        for resource_id, value in list(self.resources.items()):
            if value == resource:
                self.remove_resource_id(resource_id)

    def remove_resource_id(self, resource_id: int) -> None:
        if self.resources.pop(resource_id, None) is None:
            return

        self._prune(self.root, resource_id)

    def _prune(self, node: PredicateTreeNode, resource_id: int) -> bool:
        """Drop ``resource_id`` from the subtree, removing now-empty branches.

        Returns True once the resource was found; an id lives on exactly one
        path, so siblings are skipped after a hit.
        """
        found = False

        for child in node.predicates:
            if resource_id in child.resources:
                child.resources.discard(resource_id)
                found = True
            else:
                found = self._prune(child, resource_id)

            if found:
                break

        node.predicates[:] = [
            child for child in node.predicates if child.resources or child.predicates
        ]

        return found

    def get_resources(self, *resources) -> Iterable[_TResource]:
        return [
            self.resources[resource]
            for resource in resources
            if resource in self.resources
        ]

    def evaluate(self, *args, **kwargs) -> Iterator[int]:
        """Yield the resources on every root-to-leaf path whose predicates all
        match. All matching children are descended, not just the first."""
        stack = list(self.root.predicates)

        while stack:
            entry = stack.pop()

            if entry.predicate(*args, **kwargs):
                yield from entry.resources
                stack.extend(entry.predicates)
