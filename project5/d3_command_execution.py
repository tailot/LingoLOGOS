"""Dataset D3 generator: Multi-domain Command Execution (SFT and LLLayer).

Specifications:
- Multi-turn conversational prompts paired with executable JSON payload for multi-agent platform.
- Agents: agente_alfa, agente_beta, arbitro_logico.
- Uniform partitioning across 10 vertical domains (100,000 samples each for N = 1,000,000 total):
  1. Agenda e Scheduling Temporale
  2. Domotica e Smart Home Control
  3. Finanza Operativa e Transattiva
  4. Produttivita e Task Management
  5. Salute e Parametri Biometrici
  6. Logistica e Fleet/Supply Chain Management
  7. Gestione Middleware e System Telemetry
  8. Cybersecurity e IAM
  9. E-Commerce e Inventory Routing
  10. Customer Support e Ticket Resolution
- High-arity intents (multi-parameter with budget, scheduling, dependencies).
- Volatile context metadata.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from project5.common import DatasetSample, init_seed

AGENTS = ("agente_alfa", "agente_beta", "arbitro_logico")

DOMAINS = [
    "agenda_scheduling",
    "domotica_smart_home",
    "finanza_operativa",
    "produttivita_task",
    "salute_biometria",
    "logistica_fleet",
    "middleware_telemetry",
    "cybersecurity_iam",
    "ecommerce_inventory",
    "customer_support_tickets",
]

DOMAIN_ACTIONS = {
    "agenda_scheduling": [
        ("programma riunione urgente di allineamento", "schedule_meeting", {"priority": "high", "duration_min": 45, "requires_room": True}),
        ("riprogramma sessione di revisione sprint", "reschedule_sprint", {"priority": "medium", "duration_min": 60, "buffer_min": 15}),
    ],
    "domotica_smart_home": [
        ("imposta climatizzazione eco e spegni carichi", "set_hvac_eco", {"target_temp_c": 20.5, "shed_non_critical": True, "zone": "living"}),
        ("arma perimetrale e chiudi elettrovalvole", "arm_perimeter_safety", {"zone": "perimeter", "valves_closed": True, "alert_mode": "silent"}),
    ],
    "finanza_operativa": [
        ("esegui bonifico transattivo con verifica massimale", "execute_wire_transfer", {"currency": "EUR", "amount_limit": 5000, "aml_checked": True}),
        ("riconcilia flussi di cassa giornalieri", "reconcile_cashflow", {"ledger_id": "main_eur", "tolerance": 0.01, "audit_flag": True}),
    ],
    "produttivita_task": [
        ("assegna milestone con blocco dipendenze", "assign_milestone", {"blockers": ["ARCH-01"], "owner": "lead_dev", "sla_days": 3}),
        ("archivia sprint completato e compila retrospettiva", "close_sprint_retro", {"sprint_id": "SPRINT-42", "publish_report": True}),
    ],
    "salute_biometria": [
        ("sincronizza telemetria glicemica e frequenza cardiaca", "sync_biometrics", {"source": "ble_monitor", "heart_rate_bpm": 72, "glucose_mg_dl": 95}),
        ("invia allarme anomalia pressione sistolica", "alert_vitals_anomaly", {"systolic": 145, "diastolic": 92, "escalate_physician": True}),
    ],
    "logistica_fleet": [
        ("ottimizza rotta veicoli flotta e assegna waypoint", "optimize_fleet_route", {"fleet_size": 12, "max_payload_kg": 3500, "avoid_tolls": False}),
        ("notifica arrivo carico presso hub logistico", "log_hub_arrival", {"hub_id": "HUB_MIL_01", "dock_assigned": 4, "seal_verified": True}),
    ],
    "middleware_telemetry": [
        ("scala repliche cluster e monitora latenza P99", "autoscale_cluster", {"min_replicas": 3, "max_replicas": 10, "target_p99_ms": 120}),
        ("effettua flush buffer circolare telemetria", "flush_telemetry_buffer", {"buffer_mb": 256, "destination": "cold_storage"}),
    ],
    "cybersecurity_iam": [
        ("revoca token sessione e richiedi riautenticazione MFA", "revoke_session_tokens", {"user_uuid": "usr_99812", "force_mfa": True, "scope": "all"}),
        ("ruota certificato TLS e convalida catena di trust", "rotate_tls_cert", {"domain": "mesh.internal", "key_size": 4096, "hsm_backed": True}),
    ],
    "ecommerce_inventory": [
        ("riserva stock di magazzino e blocca carrello", "reserve_inventory_sku", {"sku": "SKU-9021", "quantity": 4, "hold_timeout_sec": 900}),
        ("aggiorna catalogo prezzi dinamici con margine minimo", "update_dynamic_pricing", {"category": "electronics", "min_margin_pct": 14.5}),
    ],
    "customer_support_tickets": [
        ("instrada ticket escalation di livello 3 a team tecnico", "escalate_ticket_l3", {"ticket_id": "TCK-88123", "severity": "P1", "notify_lead": True}),
        ("chiudi ticket con risoluzione verificata e sondaggio", "close_ticket_resolution", {"ticket_id": "TCK-88120", "resolution_code": "FIXED", "send_csat": True}),
    ],
}


@dataclass
class D3Sample(DatasetSample):
    """Schema for D3 record."""
    domain: str
    target_agent: str
    prompt: str
    dialogue_turn: int
    context_metadata: Dict[str, Any]
    execution_payload: Dict[str, Any]


class D3Generator:
    """Generator for D3 Multi-Domain Command Execution."""

    def __init__(self, seed: int = 42):
        self.rng = init_seed(seed)

    def generate_sample(self, index: int) -> D3Sample:
        # Uniform partitioning across 10 domains
        domain = DOMAINS[index % len(DOMAINS)]
        agent = AGENTS[(index // len(DOMAINS)) % len(AGENTS)]

        actions = DOMAIN_ACTIONS[domain]
        action_text, action_cmd, params = actions[(index // (len(DOMAINS) * len(AGENTS))) % len(actions)]

        volatile_clock = 1000 + (index % 50000)
        volatile_context = {
            "volatile_timestamp": volatile_clock,
            "session_id": f"sess_{(index * 31) % 100000:06d}",
            "caller_role": "system_orchestrator" if agent == "arbitro_logico" else "worker_node",
            "active_agents": list(AGENTS),
        }

        user_prompt = f"Contesto: [{domain}] Richiesta: {action_text} con priorita e routing su {agent}."

        execution_payload = {
            "routing_header": {
                "source": "orchestrator_ingress",
                "target_agent": agent,
                "command": action_cmd,
                "domain": domain,
            },
            "parameters": {
                **params,
                "request_id": f"REQ_{index:08d}",
                "execution_deadline_epoch": volatile_clock + 120,
            },
            "volatile_context": volatile_context,
            "status": "APPROVED_BY_ARBITRO" if agent == "arbitro_logico" else "PENDING_EXECUTION",
        }

        return D3Sample(
            dataset_id="D3_LLLAYER_COMMAND",
            sample_id=f"D3_{index:07d}",
            domain=domain,
            target_agent=agent,
            prompt=user_prompt,
            dialogue_turn=1 + (index % 4),
            context_metadata=volatile_context,
            execution_payload=execution_payload,
        )

    def generate_stream(self, total: int) -> Iterator[D3Sample]:
        for i in range(total):
            yield self.generate_sample(i)
