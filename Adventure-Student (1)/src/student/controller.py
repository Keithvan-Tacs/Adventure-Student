"""Pick one public action per observation by planning over the whole world.

The planner searches the state (position, collected artifacts, defeated
enemies[, time]) and minimises   r * total_cost - artifact_points,
which is exactly what the points formula rewards (total cost = accepted
actions + encounter penalties).  It therefore decides for itself whether a
reward is worth its detour and whether a blocker is cheaper to fight than to
walk around.  Patrols are handled by adding the logical turn to the state, so
waiting and timing are part of the search.
"""
from collections import deque
from math import gcd

from src.api import Position, WorldAction, WorldActionType, WorldSnapshot

_MOVES = (
    (WorldActionType.MOVE_NORTH, -1, 0),
    (WorldActionType.MOVE_SOUTH, 1, 0),
    (WorldActionType.MOVE_WEST, 0, -1),
    (WorldActionType.MOVE_EAST, 0, 1),
)


class AgentController:
    """A new controller is constructed for each run; arrival still needs EXIT."""

    def __init__(self) -> None:
        # Private planning data for this run only: a list of
        # (expected situation, action) pairs.
        self._plan = []

    #api
    def choose_action(self, state: WorldSnapshot, agent_id: str) -> WorldAction:
        key = self._situation(state)
        if not (self._plan and self._plan[0][0] == key):
            # No plan yet, or reality differs from what we expected: replan
            # from the snapshot we were actually given.
            self._plan = self._make_plan(state) or []
        if self._plan:
            _, action = self._plan.pop(0)
            return action
        return self._fallback(state)

    #sa helpers
    @staticmethod
    def _situation(state):
        pos = state.agent_position
        return ((pos.row, pos.column), state.turn,
                frozenset(state.collected_artifact_ids),
                frozenset(state.defeated_enemy_ids))

    def _make_plan(self, state):
        level = state.level
        terrain = state.terrain
        allowed = set(state.allowed_actions)
        patrols = {p.enemy_id: p for p in level.patrols}
        statics = [e for e in level.enemies if e.enemy_id not in patrols]
        artifacts = list(level.artifacts)
        rate = state.deduction_rate
        exit_pos = (level.exit.row, level.exit.column)

        static_cells = [(e.position.row, e.position.column) for e in statics]
        art_cells = [(a.position.row, a.position.column) for a in artifacts]
        collected = set(state.collected_artifact_ids)
        defeated = set(state.defeated_enemy_ids)
        cm0 = sum(1 << i for i, a in enumerate(artifacts) if a.artifact_id in collected)
        dm0 = sum(1 << i for i, e in enumerate(statics) if e.enemy_id in defeated)

        timed = bool(patrols) or state.deadline_turn is not None
        deadline = state.deadline_turn
        limit = None
        if deadline is not None:
            limit = min(deadline, state.max_turns)
        period = 1
        for p in patrols.values():
            period = period * p.period // gcd(period, p.period)

        patrol_cache = {}

        def patrol_cells(t):
            if not patrols:
                return frozenset()
            if t not in patrol_cache:
                patrol_cache[t] = frozenset(
                    (p.positions[(p.initial_phase + t) % p.period].row,
                     p.positions[(p.initial_phase + t) % p.period].column)
                    for p in patrols.values())
            return patrol_cache[t]

        def tkey(t):
            if not timed:
                return 0
            return t if deadline is not None else t % period

        start_pos = (state.agent_position.row, state.agent_position.column)
        start = (start_pos[0], start_pos[1], cm0, dm0, tkey(state.turn))
        # value = (r*cost - artifact points, actions used); smaller is better.
        best = {start: (0, 0)}
        parent = {}
        queue = deque([start])
        queued = {start}
        final = None  # (value, state before EXIT)

        def relax(src, dst, value, action):
            old = best.get(dst)
            if old is None or value < old:
                best[dst] = value
                parent[dst] = (src, action)
                if dst not in queued:
                    queued.add(dst)
                    queue.append(dst)

        while queue:
            s = queue.popleft()
            queued.discard(s)
            row, col, cm, dm, tk = s
            f, n = best[s]
            t = state.turn + n if timed else 0
            tnow = t if timed else 0
            blocked = {static_cells[i] for i in range(len(statics)) if not dm >> i & 1}
            blocked |= patrol_cells(tnow)
            next_t = tnow + 1
            next_tk = tkey(next_t) if timed else 0
            lethal_next = limit is not None and next_t >= limit
            hot_next = patrol_cells(next_t)

            def ok_after(cell):
                return not lethal_next and cell not in hot_next

            if (row, col) == exit_pos and WorldActionType.EXIT in allowed:
                total = (f + rate, n + 1)
                if final is None or total < final[0]:
                    final = (total, s)
            # movement
            for kind, dr, dc in _MOVES:
                if kind not in allowed:
                    continue
                cell = (row + dr, col + dc)
                if not terrain.is_traversable(Position(*cell)) or cell in blocked:
                    continue
                if not ok_after(cell):
                    continue
                relax(s, (cell[0], cell[1], cm, dm, next_tk), (f + rate, n + 1),
                      WorldAction(kind))
            # collect
            if WorldActionType.COLLECT in allowed and ok_after((row, col)):
                for i, a in enumerate(artifacts):
                    if not cm >> i & 1 and art_cells[i] == (row, col):
                        relax(s, (row, col, cm | 1 << i, dm, next_tk),
                              (f + rate - a.value, n + 1),
                              WorldAction(WorldActionType.COLLECT, target_id=a.artifact_id))
            # engage
            if WorldActionType.ENGAGE in allowed and ok_after((row, col)):
                for i, e in enumerate(statics):
                    cell = static_cells[i]
                    if not dm >> i & 1 and abs(cell[0] - row) + abs(cell[1] - col) == 1:
                        relax(s, (row, col, cm, dm | 1 << i, next_tk),
                              (f + rate + rate * e.encounter_penalty, n + 1),
                              WorldAction(WorldActionType.ENGAGE, target_id=e.enemy_id))
            # wait (only useful when time changes the world)
            if timed and WorldActionType.WAIT in allowed and ok_after((row, col)):
                relax(s, (row, col, cm, dm, next_tk), (f + rate, n + 1), WorldAction.wait())

        if final is None:
            return None

        # Rebuild the action sequence from the parent links.
        steps = []
        node = final[1]
        while node != start:
            node, action = parent[node][0], parent[node][1]
            steps.append(action)
        steps.reverse()
        steps.append(WorldAction(WorldActionType.EXIT))

        # Simulate it once to record the situation expected before each step.
        plan = []
        pos, turn = start_pos, state.turn
        got, beat = set(collected), set(defeated)
        for action in steps:
            plan.append(((pos, turn, frozenset(got), frozenset(beat)), action))
            kind = action.action_type
            for k, dr, dc in _MOVES:
                if kind is k:
                    pos = (pos[0] + dr, pos[1] + dc)
            if kind is WorldActionType.COLLECT:
                got.add(action.target_id)
            elif kind is WorldActionType.ENGAGE:
                beat.add(action.target_id)
            turn += 1
        return plan

    def _fallback(self, state):
        """Only used if planning finds nothing: head for the exit sensibly."""
        allowed = set(state.allowed_actions)
        here = state.agent_position
        goal = state.exit_position
        if here == goal and WorldActionType.EXIT in allowed:
            return WorldAction(WorldActionType.EXIT)
        taken = {(o.position.row, o.position.column)
                 for o in state.enemy_observations if o.position is not None}
        best, choice = None, None
        for kind, dr, dc in _MOVES:
            cell = Position(here.row + dr, here.column + dc)
            if kind not in allowed or not state.is_traversable(cell):
                continue
            if (cell.row, cell.column) in taken:
                continue
            dist = cell.manhattan_distance(goal)
            if best is None or dist < best:
                best, choice = dist, kind
        if choice is not None:
            return WorldAction(choice)
        return WorldAction.wait()
