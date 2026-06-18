"""Tests for craft.scheduling: ScheduleManager, get_schedule_from_supply."""

import numpy as np
import pytest

from craft.scheduling import ScheduleManager, get_schedule_from_supply


class TestScheduleManager:
    def test_compute_operational_times(self, timetabling):
        sm = timetabling.schedule_manager
        ot = sm.compute_operational_times()
        assert len(ot) == timetabling.n_services
        for sid, times in ot.items():
            assert isinstance(times, list)
            assert len(times) > 0
            assert all(t >= 0 for t in times)

    def test_update_from_solution_preserves_structure(self, timetabling):
        sm = timetabling.schedule_manager
        original_keys = {s: list(sm.reference_schedules[s].keys()) for s in sm.reference_schedules}
        solution = np.array(sm._get_real_vars(), dtype=np.int32)
        sm.update_from_solution(solution)
        for sid in sm.updated_schedule:
            assert list(sm.updated_schedule[sid].keys()) == original_keys[sid]

    def test_is_service_feasible_reference(self, timetabling):
        sm = timetabling.schedule_manager
        for sid in sm.reference_schedules:
            assert sm.is_service_feasible(sid) is True

    def test_get_departure_time_indexer(self, timetabling):
        idx = sm_indexer = timetabling.schedule_manager.get_departure_time_indexer()
        assert isinstance(idx, dict)
        assert all(isinstance(k, int) for k in idx)
        assert all(isinstance(v, str) for v in idx.values())

    def test_get_real_vars(self, timetabling):
        sm = timetabling.schedule_manager
        rv = sm._get_real_vars()
        assert isinstance(rv, list)
        assert len(rv) == len(sm.get_departure_time_indexer())


class TestGetScheduleFromSupply:
    def test_from_supply_object(self, supply):
        schedule = get_schedule_from_supply(supply=supply)
        assert len(schedule) == len(supply.services)
        for sid, stops in schedule.items():
            for sta, times in stops.items():
                assert len(times) == 2
                assert times[0] <= times[1]

    def test_from_path(self, supply_path):
        schedule = get_schedule_from_supply(path=supply_path)
        assert len(schedule) > 0
