"""
Load balancer interface for Artemis Router.

Sends routing decisions to a load balancer service.
"""

import time
import requests
from dataclasses import dataclass, asdict
from typing import Dict, List

from .config import LBConfig
from .schemas import RouterDecision, Sample


@dataclass
class LBMessage:
    """
    Message format for load balancer communication.

    Attributes:
        sample_id: Sample identifier
        probs: Probability distribution over models
        chosen_model: Top-1 selected model
        model_order: Order of models (for interpreting probs)
        source: Sample source
        timestamp: Unix timestamp
    """
    sample_id: str
    probs: Dict[str, float]
    chosen_model: str
    model_order: List[str]
    source: str
    timestamp: float


def send_to_lb(lb_cfg: LBConfig, decision: RouterDecision, sample: Sample):
    """
    Send router decision to load balancer.

    Args:
        lb_cfg: Load balancer configuration
        decision: Router's decision
        sample: Input sample
    """
    if not lb_cfg.enabled:
        return

    msg = LBMessage(
        sample_id=decision.sample_id,
        probs=decision.probs,
        chosen_model=decision.chosen_model,
        model_order=decision.model_order,
        source=sample.source,
        timestamp=time.time(),
    )

    if lb_cfg.protocol == "http":
        _send_http(lb_cfg, msg)
    elif lb_cfg.protocol == "kafka":
        _send_kafka(lb_cfg, msg)
    else:
        print(f"[WARN] Unknown LB protocol: {lb_cfg.protocol}")


def _send_http(lb_cfg: LBConfig, msg: LBMessage):
    """
    Send message via HTTP POST.

    Args:
        lb_cfg: Load balancer configuration
        msg: Message to send
    """
    try:
        response = requests.post(
            lb_cfg.endpoint,
            json=asdict(msg),
            timeout=0.1,  # Fast timeout to avoid blocking router
        )
        if response.status_code != 200:
            print(f"[WARN] LB returned status {response.status_code}")
    except requests.exceptions.Timeout:
        # Ignore timeouts - LB should be fast
        pass
    except Exception as e:
        # Don't crash router if LB is down
        print(f"[WARN] Failed to send to LB: {e}")


def _send_kafka(lb_cfg: LBConfig, msg: LBMessage):
    """
    Send message via Kafka (stub for future implementation).

    Args:
        lb_cfg: Load balancer configuration
        msg: Message to send
    """
    # TODO: Implement Kafka producer
    # from kafka import KafkaProducer
    # producer.send('router-decisions', value=asdict(msg))
    print("[WARN] Kafka LB not yet implemented")
