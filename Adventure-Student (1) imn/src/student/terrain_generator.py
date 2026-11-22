"""Generate canonical terrain deterministically from a specification and seed.

Every result depends only on the specification and seed (a private
``random.Random`` is built from them), so fresh instances always agree.  The
generator always aims for as many open tiles as the specification allows,
because bigger maps leave the level generator more choices.

Supported topologies
  GENERAL         a compact, connected blob of floor
  STRAIGHT_PATH   one straight line of floor
  BENT_PATH       one simple path (no touching sides) with at least one turn
  BRANCHING_TREE  a connected, cycle-free map with at least one junction
"""
import random

from src.api import GenerationFailure, Terrain, TerrainGenerationSpec, TerrainTopology

_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class TerrainGenerator:
    """Keep state local; identical inputs must also agree across fresh instances."""

    def __init__(self) -> None:
        pass

    def generate_terrain(self, specification: TerrainGenerationSpec, seed: int) -> Terrain:
        if type(specification) is not TerrainGenerationSpec:
            raise GenerationFailure("TerrainGenerator", "specification must be a TerrainGenerationSpec")
        if type(seed) is not int or seed < 0:
            raise GenerationFailure("TerrainGenerator", "seed must be a nonnegative integer")
        rows, cols = specification.rows, specification.columns
        low, high = specification.min_open_tiles, specification.max_open_tiles
        topology = specification.topology
        rng = random.Random(f"{seed}|{rows}|{cols}|{low}|{high}|{topology.value}")
        builders = {
            TerrainTopology.GENERAL: self._general,
            TerrainTopology.STRAIGHT_PATH: self._straight,
            TerrainTopology.BENT_PATH: self._bent,
            TerrainTopology.BRANCHING_TREE: self._tree,
        }
        cells = builders[topology](rows, cols, low, high, rng)
        if cells is None:
            raise GenerationFailure(
                "TerrainGenerator",
                f"no {topology.value} terrain fits {rows}x{cols} with {low}-{high} open tiles")
        return self._render(rows, cols, cells)

    @staticmethod
    def _render(rows, cols, cells):
        floor = set(cells)
        return Terrain(tuple(
            "".join("." if (r, c) in floor else "#" for c in range(cols))
            for r in range(rows)))

    @staticmethod
    def _neighbors(cell, rows, cols):
        r, c = cell
        for dr, dc in _DELTAS:
            if 0 <= r + dr < rows and 0 <= c + dc < cols:
                yield (r + dr, c + dc)

    def _general(self, rows, cols, low, high, rng):
        target = high
        if target == 0:
            return []
        cells = {(rng.randrange(rows), rng.randrange(cols))}
        while len(cells) < target:
            frontier = sorted({n for cell in cells for n in self._neighbors(cell, rows, cols)} - cells)
            # Prefer cells touching several floor tiles: compact maps with loops.
            weights = [sum(n in cells for n in self._neighbors(f, rows, cols)) ** 2 for f in frontier]
            cells.add(rng.choices(frontier, weights)[0])
        return cells

    def _straight(self, rows, cols, low, high, rng):
        length = min(high, max(rows, cols))
        if length < max(low, 2):
            return None
        options = []
        if cols >= length:
            options.append("h")
        if rows >= length:
            options.append("v")
        if rng.choice(options) == "h":
            r, c = rng.randrange(rows), rng.randrange(cols - length + 1)
            return [(r, c + i) for i in range(length)]
        r, c = rng.randrange(rows - length + 1), rng.randrange(cols)
        return [(r + i, c) for i in range(length)]

    @staticmethod
    def _serpentine(rows, cols):
        cells = []
        for k, r in enumerate(range(0, rows, 2)):
            line = range(cols) if k % 2 == 0 else range(cols - 1, -1, -1)
            cells.extend((r, c) for c in line)
            if r + 2 < rows:
                cells.append((r + 1, cols - 1 if k % 2 == 0 else 0))
        return cells

    @staticmethod
    def _is_bent_path(cells):
        floor = set(cells)
        if len(floor) != len(cells) or len(cells) < 3:
            return False
        degrees = [sum((r + dr, c + dc) in floor for dr, dc in _DELTAS) for r, c in cells]
        collinear = len({r for r, _ in cells}) == 1 or len({c for _, c in cells}) == 1
        return (not collinear and degrees.count(1) == 2
                and all(d in (1, 2) for d in degrees))

    def _transform(self, cells, rows, cols, rng):
        """Randomly mirror a pattern inside the board (keeps it an induced path)."""
        flip_r, flip_c = rng.random() < 0.5, rng.random() < 0.5
        return [((rows - 1 - r) if flip_r else r, (cols - 1 - c) if flip_c else c) for r, c in cells]

    def _bent(self, rows, cols, low, high, rng):
        pool = []
        snake = self._serpentine(rows, cols)
        pool.append(snake)
        pool.append([(c, r) for r, c in self._serpentine(cols, rows)])  # vertical snake
        for a in range(2, cols + 1):  # L shapes: along the top, then down
            pool.append([(0, c) for c in range(a)] + [(r, a - 1) for r in range(1, rows)])
        for a in range(2, rows + 1):  # L shapes: down the side, then along
            pool.append([(r, 0) for r in range(a)] + [(a - 1, c) for c in range(1, cols)])
        for _ in range(30):  # random greedy snakes
            pool.append(self._random_snake(rows, cols, rng))
        best = None
        for pattern in pool:
            m = min(high, len(pattern))
            if m < max(low, 3):
                continue
            cand = self._transform(pattern[:m], rows, cols, rng)
            if self._is_bent_path(cand) and (best is None or len(cand) > len(best)):
                best = cand
        return best

    def _random_snake(self, rows, cols, rng):
        path = [(rng.randrange(rows), rng.randrange(cols))]
        floor = set(path)
        while True:
            options = []
            for n in self._neighbors(path[-1], rows, cols):
                if n in floor:
                    continue
                if sum(m in floor for m in self._neighbors(n, rows, cols)) == 1:
                    options.append(n)
            if not options:
                return path
            nxt = rng.choice(sorted(options))
            path.append(nxt)
            floor.add(nxt)

    def _grow_tree(self, rows, cols, rng):
        """Grow one cell at a time, only onto cells touching exactly one floor tile.

        That keeps the map connected and cycle-free; every prefix of the
        returned order is itself such a tree.
        """
        start = (rng.randrange(rows), rng.randrange(cols))
        order, floor = [start], {start}
        touching = {}  # cell -> number of floor neighbours
        candidates = set()

        def add(cell):
            floor.add(cell)
            candidates.discard(cell)
            for n in self._neighbors(cell, rows, cols):
                if n in floor:
                    continue
                touching[n] = touching.get(n, 0) + 1
                if touching[n] == 1:
                    candidates.add(n)
                else:
                    candidates.discard(n)

        add(start)
        while candidates:
            nxt = rng.choice(sorted(candidates))
            order.append(nxt)
            add(nxt)
        return order

    @staticmethod
    def _comb(rows, cols):
        """Spine along the top with a tooth down every second column."""
        order = [(0, c) for c in range(cols)]
        for depth in range(1, rows):
            order.extend((depth, c) for c in range(0, cols, 2))
        return order

    @staticmethod
    def _is_branching_tree(cells):
        floor = set(cells)
        if len(floor) != len(cells) or len(cells) < 4:
            return False
        degrees = [sum((r + dr, c + dc) in floor for dr, dc in _DELTAS) for r, c in cells]
        return sum(degrees) // 2 == len(cells) - 1 and max(degrees) >= 3

    def _tree(self, rows, cols, low, high, rng):
        pool = [self._comb(rows, cols), [(c, r) for r, c in self._comb(cols, rows)]]
        for _ in range(40):
            pool.append(self._grow_tree(rows, cols, rng))
        best = None
        for pattern in pool:
            m = min(high, len(pattern))
            if m < max(low, 4):
                continue
            cand = pattern[:m]
            if self._is_branching_tree(cand) and (best is None or len(cand) > len(best)):
                best = cand
        if best is None:
            return None
        # Connectivity is guaranteed (every prefix grows from a neighbour);
        # a prefix of the comb is connected as well, and acyclic.
        return self._transform(best, rows, cols, rng)
