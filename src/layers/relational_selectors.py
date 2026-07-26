from relationship_graph import RelationshipGraph


class RelationalSelectorError(ValueError):
    """Raised when a selector requests an unknown object or relation."""


def get_object(graph, object_id):
    """Return an object by id, raising a precise error if it is absent."""
    for obj in graph["objects"]:
        if obj["id"] == object_id:
            return obj
    raise RelationalSelectorError(f"Object id {object_id} is not in the graph")


def related_objects(graph, anchor_id, relation_key, expected=True):
    """
    Return every object whose outgoing anchor relation has the expected value.

    Results are sorted by id. Ties remain candidate sets instead of being
    silently collapsed to one object.
    """
    get_object(graph, anchor_id)
    results = []

    for candidate in graph["objects"]:
        candidate_id = candidate["id"]
        if candidate_id == anchor_id:
            continue
        relation = graph["pairs"][(anchor_id, candidate_id)]
        if relation_key not in relation:
            raise RelationalSelectorError(
                f"Unknown relation key '{relation_key}'"
            )
        if relation[relation_key] == expected:
            results.append(candidate)

    return sorted(results, key=lambda obj: obj["id"])


def nearest_objects(graph, anchor_id, candidate_ids=None):
    """Return all candidates tied for minimum centroid distance."""
    return _distance_extrema(
        graph,
        anchor_id,
        candidate_ids,
        find_minimum=True,
    )


def farthest_objects(graph, anchor_id, candidate_ids=None):
    """Return all candidates tied for maximum centroid distance."""
    return _distance_extrema(
        graph,
        anchor_id,
        candidate_ids,
        find_minimum=False,
    )


def _distance_extrema(
    graph,
    anchor_id,
    candidate_ids,
    find_minimum,
):
    get_object(graph, anchor_id)
    allowed_ids = (
        set(candidate_ids)
        if candidate_ids is not None
        else {obj["id"] for obj in graph["objects"]}
    )
    allowed_ids.discard(anchor_id)

    unknown_ids = allowed_ids - {obj["id"] for obj in graph["objects"]}
    if unknown_ids:
        raise RelationalSelectorError(
            f"Candidate ids are not in graph: {sorted(unknown_ids)}"
        )
    if not allowed_ids:
        return []

    distances = {
        object_id: graph["pairs"][(anchor_id, object_id)]["centroid_distance"]
        for object_id in allowed_ids
    }
    extreme = (
        min(distances.values())
        if find_minimum
        else max(distances.values())
    )
    return [
        get_object(graph, object_id)
        for object_id in sorted(allowed_ids)
        if distances[object_id] == extreme
    ]


def unique_object_with_relation(
    graph,
    relation_key,
    expected=True,
    quantifier="any",
):
    """
    Return the sole node satisfying a relational predicate, otherwise None.

    quantifier may be "any" or "all" over the node's outgoing edges.
    """
    if quantifier not in {"any", "all"}:
        raise RelationalSelectorError(
            f"Unknown quantifier '{quantifier}'; use 'any' or 'all'"
        )

    matching = []
    for obj in graph["objects"]:
        values = []
        for other in graph["objects"]:
            if obj["id"] == other["id"]:
                continue
            relation = graph["pairs"][(obj["id"], other["id"])]
            if relation_key not in relation:
                raise RelationalSelectorError(
                    f"Unknown relation key '{relation_key}'"
                )
            values.append(relation[relation_key] == expected)

        if not values:
            continue
        qualifies = any(values) if quantifier == "any" else all(values)
        if qualifies:
            matching.append(obj)

    return matching[0] if len(matching) == 1 else None


def touching_objects(graph, anchor_id, include_diagonal=False):
    relation = "touching_8" if include_diagonal else "touching"
    return related_objects(graph, anchor_id, relation)


def aligned_objects(graph, anchor_id):
    """Return objects sharing an exact centroid row or column with anchor."""
    row_aligned = related_objects(graph, anchor_id, "same_centroid_row")
    col_aligned = related_objects(graph, anchor_id, "same_centroid_col")
    by_id = {obj["id"]: obj for obj in row_aligned + col_aligned}
    return [by_id[object_id] for object_id in sorted(by_id)]
