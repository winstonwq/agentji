#!/usr/bin/env python3
"""
agentji meta-skill script.
Reads an action + parameters from stdin as JSON, executes the action,
and writes a JSON result to stdout.

Actions:
  run            — run a named agent from a config file with a prompt
  validate       — load and validate a config file, report any errors
  list-agents    — list all agents in a config with their models and skills
  new-skill      — scaffold a SKILL.md + skill.yaml template in a target directory
  convert-skill  — generate a skill.yaml sidecar for a prompt-only skill with scripts
  serve-info     — print serve startup info for a config
"""
import json
import sys
from pathlib import Path


def _require(params: dict, *keys: str) -> None:
    missing = [k for k in keys if not params.get(k)]
    if missing:
        print(json.dumps({"error": f"Missing required parameter(s): {missing}"}))
        sys.exit(1)


# ── Actions ───────────────────────────────────────────────────────────────────

def action_run(params: dict) -> dict:
    _require(params, "config", "agent", "prompt")
    config_path = params["config"]
    agent_name = params["agent"]
    prompt = params["prompt"]

    try:
        from agentji.config import load_config
        from agentji.loop import run_agent
    except ImportError:
        return {"error": "agentji is not installed. Run: pip install agentji"}

    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        return {"error": f"Config not found: {config_path}"}
    except Exception as exc:
        return {"error": f"Config error: {exc}"}

    if agent_name not in cfg.agents:
        available = list(cfg.agents)
        return {"error": f"Agent '{agent_name}' not found. Available: {available}"}

    try:
        result = run_agent(cfg, agent_name, prompt)
        return {"success": True, "agent": agent_name, "response": result}
    except Exception as exc:
        return {"error": f"Agent run failed: {exc}"}


def action_validate(params: dict) -> dict:
    _require(params, "config")
    config_path = params["config"]

    try:
        from agentji.config import load_config
    except ImportError:
        return {"error": "agentji is not installed. Run: pip install agentji"}

    try:
        cfg = load_config(config_path)
        agents = list(cfg.agents)
        providers = list(cfg.providers)
        skills = [Path(s.path).name for s in cfg.skills]
        return {
            "valid": True,
            "config": config_path,
            "agents": agents,
            "providers": providers,
            "skills": skills,
        }
    except FileNotFoundError:
        return {"valid": False, "error": f"Config not found: {config_path}"}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def action_list_agents(params: dict) -> dict:
    _require(params, "config")
    config_path = params["config"]

    try:
        from agentji.config import load_config
    except ImportError:
        return {"error": "agentji is not installed. Run: pip install agentji"}

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        return {"error": str(exc)}

    agents = []
    for name, agent in cfg.agents.items():
        agents.append({
            "name": name,
            "model": agent.model,
            "skills": agent.skills,
            "mcps": agent.mcps,
            "builtins": agent.builtins,
            "sub_agents": agent.agents,
            "max_iterations": agent.max_iterations,
        })
    return {"config": config_path, "agents": agents}


def action_new_skill(params: dict) -> dict:
    _require(params, "skill_name", "skill_dir")
    skill_name: str = params["skill_name"]
    skill_dir: str = params["skill_dir"]

    skill_path = Path(skill_dir) / skill_name
    skill_md = skill_path / "SKILL.md"
    skill_yaml = skill_path / "skill.yaml"
    scripts_dir = skill_path / "scripts"
    run_script = scripts_dir / "run.py"

    if skill_md.exists():
        return {"error": f"Skill already exists at {skill_md}"}

    skill_path.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(exist_ok=True)

    # SKILL.md — pure Anthropic format (name + description + body only)
    skill_md.write_text(
        f"""---
name: {skill_name}
description: >
  One to three sentences describing what this skill does and when to use it.
  Be specific — the LLM uses this to decide when to call the skill.
---

# {skill_name.replace("-", " ").title()} Skill

Describe what this skill does, usage notes, and examples.
""",
        encoding="utf-8",
    )

    # skill.yaml — agentji tool config sidecar
    skill_yaml.write_text(
        f"""scripts:
  execute: scripts/run.py

parameters:
  input:
    type: string
    description: The main input for this skill.
    required: true
""",
        encoding="utf-8",
    )

    run_script.write_text(
        '''#!/usr/bin/env python3
"""
{name} skill script.
Reads parameters from stdin as JSON, writes result to stdout as JSON.
"""
import json
import sys


def main() -> None:
    try:
        params = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({{"error": f"Invalid JSON input: {{exc}}"}}))
        sys.exit(1)

    input_value = params.get("input", "")

    # TODO: implement skill logic here
    result = f"Processed: {{input_value}}"

    print(json.dumps({{"result": result}}))


if __name__ == "__main__":
    main()
'''.format(name=skill_name),
        encoding="utf-8",
    )

    return {
        "success": True,
        "skill": skill_name,
        "files_created": [str(skill_md), str(skill_yaml), str(run_script)],
        "next_steps": [
            f"Edit {skill_md} — fill in description and body (Anthropic-compatible)",
            f"Edit {skill_yaml} — define parameters and script path",
            f"Implement {run_script} — add your skill logic",
            f"Add to agentji.yaml:  skills:\\n  - path: {skill_path}",
        ],
    }


def action_convert_skill(params: dict) -> dict:
    _require(params, "skill_dir")
    skill_dir = Path(params["skill_dir"])

    if not skill_dir.exists():
        return {"error": f"Skill directory not found: {skill_dir}"}

    skill_yaml = skill_dir / "skill.yaml"
    if skill_yaml.exists():
        return {
            "info": f"skill.yaml already exists at {skill_yaml}",
            "path": str(skill_yaml),
        }

    # Determine model for the LLM call
    # Prefer explicit model param, fall back to a sensible default
    model = params.get("model") or "anthropic/claude-haiku-4-5"
    api_key = params.get("api_key") or ""

    try:
        from agentji.skill_converter import convert_skill
    except ImportError:
        return {"error": "agentji is not installed. Run: pip install agentji"}

    litellm_kwargs: dict = {"model": model}
    if api_key:
        litellm_kwargs["api_key"] = api_key

    result = convert_skill(skill_dir, litellm_kwargs)
    return result


def action_serve_info(params: dict) -> dict:
    _require(params, "config")
    config_path = params["config"]

    try:
        from agentji.config import load_config
    except ImportError:
        return {"error": "agentji is not installed. Run: pip install agentji"}

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        return {"error": str(exc)}

    default_agent = next(iter(cfg.agents))
    default_model = cfg.agents[default_agent].model
    return {
        "config": config_path,
        "default_agent": default_agent,
        "default_model": default_model,
        "agents": list(cfg.agents),
        "serve_command": f"agentji serve --config {config_path} --agent {default_agent}",
        "endpoints": {
            "chat": "POST http://localhost:8000/v1/chat/completions",
            "events": "GET  http://localhost:8000/v1/events/{run_id}",
            "pipeline": "GET  http://localhost:8000/v1/pipeline",
            "studio": "GET  http://localhost:8000/",
        },
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────

_ACTIONS = {
    "run": action_run,
    "validate": action_validate,
    "list-agents": action_list_agents,
    "new-skill": action_new_skill,
    "convert-skill": action_convert_skill,
    "serve-info": action_serve_info,
}


def main() -> None:
    try:
        params = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON input: {exc}"}))
        sys.exit(1)

    action = params.get("action", "")
    if action not in _ACTIONS:
        print(json.dumps({
            "error": f"Unknown action '{action}'. Valid: {list(_ACTIONS)}"
        }))
        sys.exit(1)

    result = _ACTIONS[action](params)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
