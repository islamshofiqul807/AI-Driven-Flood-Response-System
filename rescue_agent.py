"""
Rescue Agent
============
Automatically assigns rescue teams to distress cases when a flood is detected.
Simulates status lifecycle: standby → en_route → on_scene → returning

No LLM required – fully deterministic, reproducible with seed.
"""
from __future__ import annotations
import uuid
import datetime
import random
import math
from typing import List, Tuple, Dict

from ..schemas.models import (
    FloodEvent, DispatchPlan, DistressCase,
    RescueTeam, RescueAssignment, RescueTeamStatus,
    AgentDecision, AgentRole,
)

# ── Pre-defined rescue team roster ───────────────────────────────────────────

TEAM_ROSTER = [
    {"name": "Boat Team Alpha",        "type": "boat",       "members": 4, "speed_kmh": 25},
    {"name": "Boat Team Bravo",        "type": "boat",       "members": 4, "speed_kmh": 25},
    {"name": "Helicopter Unit 1",      "type": "helicopter", "members": 3, "speed_kmh": 120},
    {"name": "Helicopter Unit 2",      "type": "helicopter", "members": 3, "speed_kmh": 120},
    {"name": "Medical Response Alpha", "type": "medical",    "members": 5, "speed_kmh": 40},
    {"name": "Medical Response Bravo", "type": "medical",    "members": 5, "speed_kmh": 40},
    {"name": "Ground Unit 1",          "type": "ground",     "members": 6, "speed_kmh": 30},
    {"name": "Ground Unit 2",          "type": "ground",     "members": 6, "speed_kmh": 30},
]

DEPOT = {"lat": 23.8103, "lon": 90.4125}


def _haversine_km(a: Dict, b: Dict) -> float:
    R = 6371
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lon"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(h))


class RescueAgent:
    """Rescue Agent – auto-assigns rescue teams and tracks their live status."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _spawn_teams(self) -> List[RescueTeam]:
        teams = []
        for i, t in enumerate(TEAM_ROSTER):
            jlat = DEPOT["lat"] + self.rng.uniform(-0.03, 0.03)
            jlon = DEPOT["lon"] + self.rng.uniform(-0.03, 0.03)
            teams.append(RescueTeam(
                team_id=f"RSC-{i+1:02d}",
                name=t["name"],
                team_type=t["type"],
                location={"lat": round(jlat, 5), "lon": round(jlon, 5)},
                members=t["members"],
                status=RescueTeamStatus.STANDBY,
                assigned_case_id=None,
                eta_minutes=None,
            ))
        return teams

    def _calc_eta(self, team: RescueTeam, case: DistressCase) -> float:
        speed = next(t["speed_kmh"] for t in TEAM_ROSTER if t["name"] == team.name)
        dist = _haversine_km(team.location, case.location)
        return round((dist / speed) * 60 + self.rng.uniform(2, 8), 1)

    def _best_team(self, case: DistressCase, available: List[RescueTeam]) -> RescueTeam:
        if case.medical_need:
            typed = [t for t in available if t.team_type == "medical"]
        elif case.num_people > 8:
            typed = [t for t in available if t.team_type == "helicopter"]
        else:
            typed = [t for t in available if t.team_type in ("boat", "ground")]
        pool = typed if typed else available
        return min(pool, key=lambda t: _haversine_km(t.location, case.location))

    def _simulate_status(self, eta: float) -> RescueTeamStatus:
        if eta < 10:
            return RescueTeamStatus.ON_SCENE
        return RescueTeamStatus.EN_ROUTE

    def assign(
        self,
        flood_event: FloodEvent,
        plan: DispatchPlan,
    ) -> Tuple[List[RescueTeam], List[RescueAssignment], AgentDecision]:
        all_teams = self._spawn_teams()

        if not flood_event.active:
            decision = AgentDecision(
                agent=AgentRole.RESCUE,
                timestamp=datetime.datetime.now(),
                input_summary="Flood event is not active.",
                output_summary="No rescue teams dispatched — flood not confirmed.",
                reasoning=(
                    "Flood confidence is below activation threshold. "
                    "All rescue teams remain on standby at depot. Monitoring continues."
                ),
                data={"dispatched": 0, "standby": len(all_teams)},
            )
            return all_teams, [], decision

        priority_cases = sorted(plan.cases, key=lambda c: c.urgency_score, reverse=True)
        available = list(all_teams)
        assignments: List[RescueAssignment] = []

        for case in priority_cases:
            if not available:
                break
            team = self._best_team(case, available)
            eta = self._calc_eta(team, case)
            status = self._simulate_status(eta)

            team.status = status
            team.assigned_case_id = case.case_id
            team.eta_minutes = eta
            available.remove(team)

            notes_parts = []
            if case.medical_need:
                notes_parts.append("Medical priority case")
            notes_parts.append(f"ETA {eta} min")
            notes_parts.append(f"{case.num_people} people at location")
            notes_parts.append(f"Urgency {case.urgency_score}/10")

            assignments.append(RescueAssignment(
                assignment_id=f"ASGN-{uuid.uuid4().hex[:6].upper()}",
                team=team,
                case_id=case.case_id,
                case_location=case.location,
                urgency_score=case.urgency_score,
                dispatched_at=datetime.datetime.now(),
                status=status,
                notes=" · ".join(notes_parts),
            ))

        on_scene = sum(1 for a in assignments if a.status == RescueTeamStatus.ON_SCENE)
        en_route = sum(1 for a in assignments if a.status == RescueTeamStatus.EN_ROUTE)
        standby  = len(available)
        med_asgn = sum(1 for a in assignments if a.team.team_type == "medical")

        decision = AgentDecision(
            agent=AgentRole.RESCUE,
            timestamp=datetime.datetime.now(),
            input_summary=(
                f"{len(plan.cases)} distress cases · {len(all_teams)} rescue teams available."
            ),
            output_summary=(
                f"{len(assignments)} teams dispatched — "
                f"{on_scene} on scene, {en_route} en route, {standby} on standby."
            ),
            reasoning=(
                f"Rescue agent automatically assigned {len(assignments)} teams to the "
                f"{len(priority_cases)} highest-priority distress cases. "
                f"Teams selected by type match (medical/boat/helicopter) and proximity. "
                f"{med_asgn} medical-specialist teams assigned to medical-priority cases. "
                f"{standby} teams held in reserve for secondary response waves."
            ),
            data={
                "total_teams": len(all_teams),
                "dispatched":  len(assignments),
                "on_scene":    on_scene,
                "en_route":    en_route,
                "standby":     standby,
                "medical_assignments": med_asgn,
            },
        )

        return all_teams, assignments, decision
