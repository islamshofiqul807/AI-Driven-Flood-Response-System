"""
Test suite for the AI Disaster Response System.

Run with: python -m pytest tests/ -v
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.synthetic.engine import SyntheticDataEngine, SCENARIO_CONFIGS
from backend.detection.flood_detector import FloodDetector
from backend.planning.engine import PlanningEngine, haversine_km
from backend.agents.orchestrator import MultiAgentOrchestrator
from backend.evaluation.evaluator import EvaluationEngine
from backend.simulation.runner import SimulationRunner
from backend.schemas.models import Severity, ResourceType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def s3_data():
    eng = SyntheticDataEngine("S3", seed=42)
    return {
        "weather":   eng.generate_weather_stream(),
        "satellite": eng.generate_satellite_polygons(),
        "reports":   eng.generate_distress_reports(),
        "truth":     eng.generate_ground_truth(),
    }

@pytest.fixture(scope="module")
def s3_result():
    return SimulationRunner().run("S3", seed=42)


# ── Synthetic Engine Tests ─────────────────────────────────────────────────────

class TestSyntheticEngine:

    def test_all_scenarios_available(self):
        assert set(SCENARIO_CONFIGS.keys()) == {"S1","S2","S3","S4","S5","S6"}

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError):
            SyntheticDataEngine("S99")

    def test_weather_stream_count(self):
        eng = SyntheticDataEngine("S3", seed=0)
        w = eng.generate_weather_stream(num_readings=6)
        assert len(w) == 6

    def test_weather_deterministic(self):
        w1 = SyntheticDataEngine("S3", seed=42).generate_weather_stream()
        w2 = SyntheticDataEngine("S3", seed=42).generate_weather_stream()
        assert w1[0].rainfall_mm_per_hr == w2[0].rainfall_mm_per_hr

    def test_different_seeds_differ(self):
        w1 = SyntheticDataEngine("S3", seed=1).generate_weather_stream()
        w2 = SyntheticDataEngine("S3", seed=2).generate_weather_stream()
        assert w1[0].rainfall_mm_per_hr != w2[0].rainfall_mm_per_hr

    def test_s1_low_rainfall(self):
        eng = SyntheticDataEngine("S1", seed=42)
        w = eng.generate_weather_stream()
        assert all(r.rainfall_mm_per_hr <= 10 for r in w)

    def test_s2_high_rainfall(self):
        eng = SyntheticDataEngine("S2", seed=42)
        w = eng.generate_weather_stream()
        assert all(r.rainfall_mm_per_hr >= 60 for r in w)

    def test_s6_no_satellite(self):
        eng = SyntheticDataEngine("S6", seed=42)
        assert eng.generate_satellite_polygons() == []

    def test_distress_report_ids_unique(self):
        reports = SyntheticDataEngine("S2", seed=42).generate_distress_reports()
        ids = [r.report_id for r in reports]
        assert len(ids) == len(set(ids))

    def test_s4_low_severity(self):
        reports = SyntheticDataEngine("S4", seed=42).generate_distress_reports()
        assert all(r.severity_estimate <= 2 for r in reports)

    def test_ground_truth_s1_no_flood(self):
        truth = SyntheticDataEngine("S1", seed=42).generate_ground_truth()
        assert truth["is_real_flood"] is False

    def test_ground_truth_s2_real_flood(self):
        truth = SyntheticDataEngine("S2", seed=42).generate_ground_truth()
        assert truth["is_real_flood"] is True


# ── Detection Tests ────────────────────────────────────────────────────────────

class TestFloodDetector:

    def test_s2_detects_severe(self, s3_data):
        eng = SyntheticDataEngine("S2", seed=42)
        det = FloodDetector("S2")
        fe = det.detect(
            eng.generate_weather_stream(),
            eng.generate_satellite_polygons(),
            eng.generate_distress_reports(),
        )
        assert fe.severity in (Severity.SEVERE, Severity.EXTREME)
        assert fe.confidence > 0.7

    def test_s1_low_confidence(self):
        eng = SyntheticDataEngine("S1", seed=42)
        det = FloodDetector("S1")
        fe = det.detect(
            eng.generate_weather_stream(),
            eng.generate_satellite_polygons(),
            eng.generate_distress_reports(),
        )
        assert fe.confidence < 0.4

    def test_s6_no_satellite_reduces_confidence(self):
        eng = SyntheticDataEngine("S6", seed=42)
        det = FloodDetector("S6")
        fe = det.detect(eng.generate_weather_stream(), [], eng.generate_distress_reports())
        # S3 equivalent with satellite
        eng3 = SyntheticDataEngine("S3", seed=42)
        det3 = FloodDetector("S3")
        fe3 = det3.detect(eng3.generate_weather_stream(), eng3.generate_satellite_polygons(), eng3.generate_distress_reports())
        assert fe.confidence < fe3.confidence

    def test_geojson_valid_structure(self, s3_data):
        det = FloodDetector("S3")
        fe = det.detect(s3_data["weather"], s3_data["satellite"], s3_data["reports"])
        gj = fe.affected_area_geojson
        assert gj["type"] == "FeatureCollection"
        assert isinstance(gj["features"], list)
        assert len(gj["features"]) > 0

    def test_event_id_format(self, s3_data):
        det = FloodDetector("S3")
        fe = det.detect(s3_data["weather"], s3_data["satellite"], s3_data["reports"])
        assert fe.event_id.startswith("EVT-S3-")

    def test_estimated_people_positive(self, s3_data):
        det = FloodDetector("S3")
        fe = det.detect(s3_data["weather"], s3_data["satellite"], s3_data["reports"])
        assert fe.estimated_affected_people > 0

    def test_evidence_has_three_items(self, s3_data):
        det = FloodDetector("S3")
        fe = det.detect(s3_data["weather"], s3_data["satellite"], s3_data["reports"])
        assert len(fe.evidence) == 3

    def test_no_data_no_flood(self):
        det = FloodDetector("S1")
        fe = det.detect([], [], [])
        assert fe.confidence == 0.0
        assert fe.active is False


# ── Planning Engine Tests ──────────────────────────────────────────────────────

class TestPlanningEngine:

    def test_haversine_same_point(self):
        loc = {"lat": 23.8, "lon": 90.4}
        assert haversine_km(loc, loc) == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        dhaka = {"lat": 23.8103, "lon": 90.4125}
        ctg   = {"lat": 22.3569, "lon": 91.7832}
        d = haversine_km(dhaka, ctg)
        assert 180 < d < 220  # ~198 km

    def test_plan_has_all_components(self, s3_result):
        plan = s3_result.dispatch_plan
        assert plan.total_cases > 0
        assert len(plan.resources) > 0
        assert len(plan.shelters) > 0

    def test_cases_sorted_by_urgency(self, s3_result):
        cases = s3_result.dispatch_plan.cases
        urgencies = [c.urgency_score for c in cases]
        assert urgencies == sorted(urgencies, reverse=True)

    def test_all_cases_have_cluster(self, s3_result):
        for c in s3_result.dispatch_plan.cases:
            assert c.cluster_id is not None

    def test_resource_types_present(self, s3_result):
        types = {r.resource_type for r in s3_result.dispatch_plan.resources}
        assert ResourceType.RESCUE_BOAT in types
        assert ResourceType.HELICOPTER in types

    def test_shelters_have_capacity(self, s3_result):
        for s in s3_result.dispatch_plan.shelters:
            assert s.capacity > 0

    def test_high_urgency_cases_assigned(self, s3_result):
        high = [c for c in s3_result.dispatch_plan.cases if c.urgency_score >= 6.0]
        assigned = [c for c in high if c.assigned_resource]
        assert len(assigned) == len(high)

    def test_medical_cases_get_ambulance_or_helicopter(self, s3_result):
        medical = [c for c in s3_result.dispatch_plan.cases if c.medical_need]
        for c in medical:
            if c.assigned_resource:
                assert any(
                    c.assigned_resource.startswith(p)
                    for p in ("AMBULANCE", "HELICOPTER", "RESCUE_BOAT")
                )

    def test_routes_have_valid_hazard(self, s3_result):
        for r in s3_result.dispatch_plan.routes:
            assert 0.0 <= r.hazard_level <= 1.0


# ── Agent Tests ────────────────────────────────────────────────────────────────

class TestAgentOrchestrator:

    def test_sitrep_has_six_agents(self, s3_result):
        # Now 7 agents: triage, cluster, resource, routing, medical, rescue, command
        assert len(s3_result.sitrep.agent_decisions) == 7

    def test_all_agent_roles_present(self, s3_result):
        from backend.schemas.models import AgentRole
        roles = {d.agent for d in s3_result.sitrep.agent_decisions}
        assert AgentRole.RESCUE in roles
        assert AgentRole.COMMAND in roles
        assert AgentRole.TRIAGE in roles

    def test_sitrep_narrative_nonempty(self, s3_result):
        assert len(s3_result.sitrep.narrative) > 20

    def test_sitrep_has_recommendations(self, s3_result):
        assert len(s3_result.sitrep.recommendations) >= 3

    def test_active_flood_sitrep_status(self, s3_result):
        assert s3_result.sitrep.status == "ACTIVE"

    def test_s1_resolved_status(self):
        result = SimulationRunner().run("S1", seed=42)
        assert result.sitrep.status == "RESOLVED"


# ── Evaluation Tests ───────────────────────────────────────────────────────────

class TestEvaluationEngine:

    def test_metrics_range_valid(self, s3_result):
        m = s3_result.metrics
        assert 0 <= m.detection_precision <= 1
        assert 0 <= m.detection_recall <= 1
        assert 0 <= m.urgent_case_coverage_pct <= 100
        assert m.avg_response_time_min > 0
        assert 0 <= m.resource_utilization_rate <= 100
        assert 0 <= m.route_safety_score <= 1
        assert 0 <= m.fairness_distribution_score <= 1
        assert 0 <= m.overall_score <= 100

    def test_s2_high_overall_score(self):
        result = SimulationRunner().run("S2", seed=42)
        assert result.metrics.overall_score > 70

    def test_s1_no_flood_correct_rejection(self):
        result = SimulationRunner().run("S1", seed=42)
        # S1 is a correct non-detection (low confidence, not active)
        # precision=1.0 and recall=1.0 because we correctly didn't detect a real flood
        assert result.flood_event.active is False
        assert result.flood_event.confidence < 0.4
        assert result.metrics.overall_score > 50  # still a valid response (correct rejection)

    def test_all_scenarios_produce_metrics(self):
        runner = SimulationRunner()
        for sc in SCENARIO_CONFIGS:
            result = runner.run(sc, seed=42)
            assert result.metrics.overall_score > 0


# ── Integration / End-to-End Tests ────────────────────────────────────────────

class TestEndToEnd:

    def test_full_pipeline_s3(self, s3_result):
        assert s3_result.flood_event is not None
        assert s3_result.dispatch_plan is not None
        assert s3_result.sitrep is not None
        assert s3_result.metrics is not None

    def test_to_dict_serialisable(self, s3_result):
        import json
        d = s3_result.to_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100

    def test_to_json_valid(self, s3_result):
        import json
        parsed = json.loads(s3_result.to_json())
        assert "flood_event" in parsed
        assert "dispatch_plan" in parsed
        assert "sitrep" in parsed
        assert "metrics" in parsed

    def test_reproducibility(self):
        r1 = SimulationRunner().run("S3", seed=42)
        r2 = SimulationRunner().run("S3", seed=42)
        assert r1.flood_event.event_id != r2.flood_event.event_id  # UUID differs
        assert r1.metrics.overall_score == r2.metrics.overall_score  # metrics identical

    def test_different_seeds_different_results(self):
        r1 = SimulationRunner().run("S3", seed=1)
        r2 = SimulationRunner().run("S3", seed=2)
        assert r1.flood_event.confidence != r2.flood_event.confidence

    def test_timeline_populated(self, s3_result):
        # 6 steps: data, detection, plan, agents, rescue, evaluation
        assert len(s3_result.timeline) == 6

    @pytest.mark.parametrize("scenario", list(SCENARIO_CONFIGS.keys()))
    def test_all_scenarios_complete(self, scenario):
        result = SimulationRunner().run(scenario, seed=42)
        assert result.metrics.overall_score >= 0


# ── Rescue Agent Tests ─────────────────────────────────────────────────────────

class TestRescueAgent:

    def test_rescue_teams_present_in_sitrep(self, s3_result):
        assert isinstance(s3_result.sitrep.rescue_teams, list)
        assert len(s3_result.sitrep.rescue_teams) == 8

    def test_rescue_dispatched_when_flood_active(self, s3_result):
        assert s3_result.flood_event.active is True
        assert len(s3_result.sitrep.rescue_assignments) > 0

    def test_s1_no_rescue_dispatch(self):
        r = SimulationRunner().run("S1", seed=42)
        assert r.flood_event.active is False
        assert len(r.sitrep.rescue_assignments) == 0

    def test_s2_all_teams_dispatched(self):
        r = SimulationRunner().run("S2", seed=42)
        assert len(r.sitrep.rescue_assignments) == 8  # all 8 teams dispatched

    def test_medical_cases_get_medical_teams(self):
        r = SimulationRunner().run("S5", seed=42)  # medical-heavy
        medical_asgns = [
            a for a in r.sitrep.rescue_assignments
            if a["team"]["team_type"] == "medical"
        ]
        assert len(medical_asgns) >= 1

    def test_rescue_status_valid_values(self, s3_result):
        valid = {"standby", "en_route", "on_scene", "returning"}
        for a in s3_result.sitrep.rescue_assignments:
            assert a["status"] in valid

    def test_rescue_eta_positive(self, s3_result):
        for a in s3_result.sitrep.rescue_assignments:
            eta = a["team"].get("eta_minutes")
            if eta is not None:
                assert eta > 0

    def test_rescue_assignment_ids_unique(self, s3_result):
        ids = [a["assignment_id"] for a in s3_result.sitrep.rescue_assignments]
        assert len(ids) == len(set(ids))

    def test_rescue_summary_method(self, s3_result):
        summary = s3_result.rescue_summary()
        assert "total_teams" in summary
        assert "dispatched" in summary
        assert "coverage_pct" in summary
        assert 0 <= summary["coverage_pct"] <= 100

    def test_rescue_agent_in_agent_decisions(self, s3_result):
        agents = [d.agent.value for d in s3_result.sitrep.agent_decisions]
        assert "rescue" in agents

    def test_on_scene_teams_have_low_eta(self, s3_result):
        for a in s3_result.sitrep.rescue_assignments:
            if a["status"] == "on_scene":
                assert a["team"]["eta_minutes"] < 10
