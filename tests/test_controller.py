import math

import pytest

from experiment import ControllerObservation, LOCCOConfig, LOCCOController


def make_controller() -> LOCCOController:
    return LOCCOController(
        LOCCOConfig(
            gamma_tgt=0.2,
            rho=0.9,
            mu_B=0.9,
            deadband=0.02,
            batch_gain=0.05,
            learning_rate_gain=0.01,
            Ls=1.0,
            B_min=8,
            B_max=512,
            eta_min=1e-4,
            eta_max=0.05,
        ),
        eta0=0.01,
        batch0=64,
    )


def observation(loss_after: float) -> ControllerObservation:
    return ControllerObservation(
        loss_before=1.0,
        loss_after=loss_after,
        grad_norm2=1.0,
        eta=0.01,
        batch=64,
    )


def test_positive_error_increases_batch_and_decreases_learning_rate() -> None:
    controller = make_controller()

    decision = None
    for _ in range(20):
        decision = controller.update(observation(loss_after=1.0))

    assert decision is not None
    assert decision.error > 0.0
    assert decision.batch > 64
    assert decision.eta < 0.01


def test_negative_error_decreases_batch_and_increases_learning_rate() -> None:
    controller = make_controller()

    decision = controller.update(observation(loss_after=0.8))

    assert decision.error < 0.0
    assert decision.batch < 64
    assert decision.eta > 0.01


def test_deadband_suppresses_control_update() -> None:
    controller = make_controller()

    delta_det = 0.01 - 0.5 * 0.01**2
    target_observed_drop = delta_det - 0.2 * 0.01
    decision = controller.update(observation(loss_after=1.0 - target_observed_drop))

    assert math.isclose(decision.error, 0.0, abs_tol=1e-12)
    assert decision.batch == 64
    assert math.isclose(decision.eta, 0.01, rel_tol=0.0, abs_tol=1e-12)


def test_controller_clips_to_configured_bounds() -> None:
    controller = LOCCOController(
        LOCCOConfig(
            gamma_tgt=0.2,
            batch_gain=100.0,
            learning_rate_gain=100.0,
            B_min=32,
            B_max=64,
            eta_min=0.005,
            eta_max=0.02,
        ),
        eta0=0.01,
        batch0=32,
    )

    decision = controller.update(observation(loss_after=1.0))

    assert 32 <= decision.batch <= 64
    assert 0.005 <= decision.eta <= 0.02


def test_non_finite_observation_fails_loudly() -> None:
    controller = make_controller()
    bad = ControllerObservation(
        loss_before=float("nan"),
        loss_after=1.0,
        grad_norm2=1.0,
        eta=0.01,
        batch=64,
    )

    with pytest.raises(ValueError, match="non-finite"):
        controller.update(bad)
