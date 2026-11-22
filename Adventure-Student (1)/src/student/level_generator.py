"""Build a complete initial world, using the injected terrain generator.

For each attempt the generator asks the supplied terrain dependency for a map
(seed, seed+1, ... as the dependency rules allow), checks that the map can
satisfy the request, picks distinct start/exit tiles that are as far apart as
the specification allows, then places the requested artifacts and enemies.
It returns an initial world only: no route and no score.
"""
import random
from collections import deque

from src.api import (
    GenerationFailure, LevelDefinition, LevelSpecification, Position,
    TerrainGeneratorContract, TerrainTopology,
    to_artifact_definition, to_character_definition, to_enemy_definition,
)
from src.student.artifact import Artifact
from src.student.character import Character
from src.student.enemy import Enemy

_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MAX_SEED = 2 ** 31 - 1


class LevelGenerator:
    """Use the supplied TerrainGeneratorContract; do not instantiate a replacement."""

    def __init__(self, terrain_generator: TerrainGeneratorContract) -> None:
        self._terrain_generator = terrain_generator

    # ------------------------------------------------------------------ API
    def generate_level(self, specification: LevelSpecification, seed: int) -> LevelDefinition:
        if type(specification) is not LevelSpecification:
            raise GenerationFailure("LevelGenerator", "specification must be a LevelSpecification")
        if type(seed) is not int or seed < 0:
            raise GenerationFailure("LevelGenerator", "seed must be a nonnegative integer")
        reason = "no attempt was made"
        for attempt in range(specification.max_attempts):
            terrain_seed = seed + attempt  # first call uses the seed itself
            if terrain_seed > _MAX_SEED:
                break
            try:
                terrain = self._terrain_generator.generate_terrain(
                    specification.terrain_spec, terrain_seed)
            except GenerationFailure as failure:
                reason = failure.reason
                continue
            level, reason = self._assemble(specification, seed, terrain)
            if level is not None:
                return level
        raise GenerationFailure("LevelGenerator", reason)

    # ------------------------------------------------------------- assembly
    def _assemble(self, spec, seed, terrain):
        floor = {(r, c) for r, row in enumerate(terrain.terrain_rows)
                 for c, ch in enumerate(row) if ch == "."}
        if len(floor) < 2:
            return None, "terrain has fewer than two floor tiles"
        graph = {cell: [(cell[0] + dr, cell[1] + dc) for dr, dc in _DELTAS
                        if (cell[0] + dr, cell[1] + dc) in floor] for cell in floor}
        branches = sum(len(n) >= 3 for n in graph.values())
        dead_ends = sum(len(n) == 1 for n in graph.values())
        if branches < spec.min_branch_tiles:
            return None, "terrain has too few branch tiles"
        if dead_ends > spec.max_dead_end_tiles:
            return None, "terrain has too many dead ends"

        rng = random.Random(f"{seed}|{spec.level_id}|{spec.profile_id}|level")
        endpoints = self._choose_endpoints(spec, graph, rng)
        if endpoints is None:
            return None, "no start/exit pair meets the route requirements"
        start, goal = endpoints

        placed = self._place(spec, sorted(floor), start, goal, rng)
        if placed is None:
            return None, "not enough floor tiles to place the requested entities"
        artifacts, enemies = placed

        return LevelDefinition(
            level_id=spec.level_id, profile_id=spec.profile_id, seed=seed,
            terrain=terrain, start=Position(*start), exit=Position(*goal),
            character=to_character_definition(Character(spec.agent_id)),
            artifacts=tuple(artifacts), enemies=tuple(enemies), patrols=(),
        ), ""

    # ------------------------------------------------------------ endpoints
    def _choose_endpoints(self, spec, graph, rng):
        topology = spec.terrain_spec.topology
        leaves_only = topology in (TerrainTopology.STRAIGHT_PATH,
                                   TerrainTopology.BENT_PATH,
                                   TerrainTopology.BRANCHING_TREE)
        cells = sorted(graph)
        candidates = []
        for start in cells:
            if leaves_only and len(graph[start]) != 1:
                continue
            for goal, dist in self._distances(graph, start).items():
                if goal == start or (leaves_only and len(graph[goal]) != 1):
                    continue
                if not spec.min_route_moves <= dist <= spec.max_route_moves:
                    continue
                if dist + 1 > spec.max_turns:  # moves plus the final EXIT
                    continue
                candidates.append((-dist, rng.random(), start, goal))
        candidates.sort()
        for _, _, start, goal in candidates:
            if not spec.require_alternate_path or self._has_alternate_path(graph, start, goal):
                return start, goal
        return None

    @staticmethod
    def _distances(graph, start):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for n in graph[cell]:
                if n not in dist:
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        return dist

    @staticmethod
    def _has_alternate_path(graph, start, goal):
        """True when some edge of a start-goal path is not a bridge."""
        disc, low, parent = {start: 0}, {start: 0}, {start: None}
        bridges = set()
        stack = [(start, iter(graph[start]))]
        while stack:
            vertex, neighbours = stack[-1]
            n = next(neighbours, None)
            if n is not None:
                if n == parent[vertex]:
                    continue
                if n in disc:
                    low[vertex] = min(low[vertex], disc[n])
                else:
                    parent[n] = vertex
                    disc[n] = low[n] = len(disc)
                    stack.append((n, iter(graph[n])))
            else:
                stack.pop()
                p = parent[vertex]
                if p is not None:
                    if low[vertex] > disc[p]:
                        bridges.add((p, vertex))
                    low[p] = min(low[p], low[vertex])
        vertex = goal
        while vertex != start:
            p = parent[vertex]
            if (p, vertex) not in bridges:
                return True
            vertex = p
        return False

    # ------------------------------------------------------------ placement
    def _place(self, spec, floor, start, goal, rng):
        endpoints = {start, goal}
        enemy_pool = [c for c in floor if c != start
                      and (spec.allow_entities_on_endpoints or c not in endpoints)]
        artifact_pool = [c for c in floor
                         if spec.allow_entities_on_endpoints or c not in endpoints]
        a_count, e_count = spec.artifact_count, spec.enemy_count
        if spec.allow_entity_overlap:
            if (e_count and not enemy_pool) or (a_count and not artifact_pool):
                return None
            enemy_cells = [rng.choice(enemy_pool) for _ in range(e_count)]
            artifact_cells = [rng.choice(artifact_pool) for _ in range(a_count)]
        else:
            if len(enemy_pool) < e_count or len(artifact_pool) < a_count + e_count:
                return None
            enemy_cells = rng.sample(enemy_pool, e_count)
            left = [c for c in artifact_pool if c not in set(enemy_cells)]
            artifact_cells = rng.sample(left, a_count)
        artifacts = [
            to_artifact_definition(Artifact(
                f"artifact-{i}", Position(*cell),
                rng.randint(spec.artifact_value_min, spec.artifact_value_max)))
            for i, cell in enumerate(artifact_cells)]
        enemies = [
            to_enemy_definition(Enemy(
                f"enemy-{i}", Position(*cell),
                rng.randint(spec.enemy_penalty_min, spec.enemy_penalty_max)))
            for i, cell in enumerate(enemy_cells)]
        return artifacts, enemies
