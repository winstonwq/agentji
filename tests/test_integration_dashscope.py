"""Integration tests against the real DashScope (Qwen) API.

These tests make real network calls and consume API credits.
Run with: pytest -m integration

Requires DASHSCOPE_API_KEY set in .env or environment.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from agentji.config import load_config
from agentji.loop import run_agent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_test_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "agentji.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── Basic LLM call (no tools) ─────────────────────────────────────────────────

@pytest.mark.integration
class TestDashScopeBasic:
    def test_simple_prompt_returns_response(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """A simple prompt should return a non-empty string."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              qa:
                model: qwen/qwen-plus
                system_prompt: "You are a helpful assistant. Be concise."
                max_iterations: 3
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "qa", "Reply with exactly: AGENTJI_OK")
        assert "AGENTJI_OK" in result

    def test_chinese_prompt(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """Qwen should handle Chinese input correctly."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              qa:
                model: qwen/qwen-plus
                system_prompt: "你是一个助手。请用中文简洁回答。"
                max_iterations: 3
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "qa", "你好，请用一句话介绍agentji")
        assert len(result) > 5

    def test_qwen_max_reasoning(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """qwen-max should handle a basic reasoning task."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              reasoner:
                model: qwen/qwen-max
                system_prompt: "You are a precise assistant. Answer only with the number."
                max_iterations: 3
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "reasoner", "What is 17 multiplied by 13? Answer with only the number.")
        assert "221" in result


# ── Tool use with hello-world skill ──────────────────────────────────────────

@pytest.mark.integration
class TestDashScopeWithSkill:
    def test_hello_world_skill_called(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """The agent should call the hello-world skill when asked."""
        skills_root = Path(__file__).parent.parent / "skills"
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            skills:
              - path: {skills_root / "hello-world"}
            agents:
              greeter:
                model: qwen/qwen-plus
                system_prompt: "You must use the hello-world tool to greet the user."
                skills: [hello-world]
                max_iterations: 5
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "greeter", "Use the hello-world tool to greet Winston")
        assert "Winston" in result


# ── MCP integration: Yahoo Finance ───────────────────────────────────────────

@pytest.mark.integration
class TestDashScopeWithYahooFinanceMcp:
    def test_data_fetcher_calls_mcp_tool(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """data-fetcher agent should call Yahoo Finance MCP and return financial data."""
        mcp_server = Path(__file__).parent.parent / "examples" / "ftse-analysis" / "yahoo_finance_mcp.py"
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            mcps:
              - name: yahoo-finance
                command: python
                args: ["{mcp_server}"]
            agents:
              data-fetcher:
                model: qwen/qwen-plus
                system_prompt: >
                  You are a financial data agent.
                  Always call get_key_metrics to retrieve data.
                  Output the raw result verbatim.
                mcps: [yahoo-finance]
                max_iterations: 5
        """)
        cfg = load_config(config_path)
        result = run_agent(
            cfg,
            "data-fetcher",
            "Get key metrics for AZN.L (AstraZeneca)",
        )
        # Should contain some financial data markers
        assert len(result) > 100
        # Should mention AstraZeneca or AZN
        assert any(kw in result.upper() for kw in ["AZN", "ASTRAZENECA", "MARKET_CAP", "P/E", "PE"])

    def test_full_two_agent_pipeline(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """Full pipeline: data-fetcher fetches, analyst produces a report."""
        mcp_server = Path(__file__).parent.parent / "examples" / "ftse-analysis" / "yahoo_finance_mcp.py"
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            mcps:
              - name: yahoo-finance
                command: python
                args: ["{mcp_server}"]
            agents:
              data-fetcher:
                model: qwen/qwen-plus
                system_prompt: >
                  You are a financial data agent.
                  Call get_key_metrics and output the raw result.
                mcps: [yahoo-finance]
                max_iterations: 5
              analyst:
                model: qwen/qwen-max
                system_prompt: >
                  You are a financial analyst.
                  Given financial data, write a brief 3-sentence investment summary
                  covering valuation, profitability, and a verdict (Buy/Hold/Sell).
                max_iterations: 3
        """)
        cfg = load_config(config_path)

        # Step 1: fetch data
        raw_data = run_agent(
            cfg,
            "data-fetcher",
            "Get key metrics for AZN.L",
        )
        assert len(raw_data) > 50

        # Step 2: analyse
        analysis = run_agent(
            cfg,
            "analyst",
            raw_data,
        )
        assert len(analysis) > 50
        # Should contain a verdict
        assert any(v in analysis.upper() for v in ["BUY", "HOLD", "SELL"])


# ── Visualizer: Wanx image generation ────────────────────────────────────────

@pytest.mark.integration
class TestDashScopeVisualizer:
    def test_visualizer_generates_and_saves_image(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """Full three-agent pipeline: fetch → analyse → visualize (real Wanx call)."""
        mcp_server = Path(__file__).parent.parent / "examples" / "ftse-analysis" / "yahoo_finance_mcp.py"
        wan_skill = Path(__file__).parent.parent / "skills" / "wan-image"
        output_image = tmp_path / "test_infographic.png"

        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            skills:
              - path: {wan_skill}
            mcps:
              - name: yahoo-finance
                command: python
                args: ["{mcp_server}"]
            agents:
              data-fetcher:
                model: qwen/qwen-plus
                system_prompt: >
                  You are a financial data agent.
                  Call get_key_metrics and output the raw result.
                mcps: [yahoo-finance]
                max_iterations: 5
              analyst:
                model: qwen/qwen-max
                system_prompt: >
                  You are a financial analyst.
                  Given financial data, write a concise investment summary
                  covering: company name, sector, top 3 metrics, verdict (Buy/Hold/Sell).
                  Keep it under 200 words.
                max_iterations: 3
              visualizer:
                model: qwen/qwen-max
                system_prompt: >
                  You are a financial infographics designer.
                  Read the analysis and call wan-image to generate an infographic.
                  Craft a prompt describing: company name, sector, key metric directions,
                  verdict badge, dark background, Bloomberg-style, professional data visualization.
                  Do NOT include specific numbers in the prompt — describe visual direction only.
                  Use output_path: {output_image}
                  Use size: 1280*720
                  Use model: wanx2.1-t2i-plus
                skills: [wan-image]
                max_iterations: 5
        """)
        cfg = load_config(config_path)

        # Step 1: fetch
        raw_data = run_agent(cfg, "data-fetcher", "Get key metrics for AZN.L")
        assert len(raw_data) > 50

        # Step 2: analyse
        analysis = run_agent(cfg, "analyst", raw_data)
        assert len(analysis) > 50

        # Step 3: visualize — real Wanx API call (~30-60s)
        viz_result = run_agent(cfg, "visualizer", analysis)
        assert len(viz_result) > 10

        # Image should be saved on disk
        assert output_image.exists(), f"Expected image at {output_image} but it was not created"
        assert output_image.stat().st_size > 1000, "Image file is suspiciously small"

    def test_visualizer_standalone_with_canned_analysis(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """Run only the visualizer with a handcrafted analysis — faster and cheaper."""
        wan_skill = Path(__file__).parent.parent / "skills" / "wan-image"
        output_image = tmp_path / "standalone_infographic.png"

        canned_analysis = textwrap.dedent(f"""
            ## AstraZeneca (AZN.L) — Investment Analysis

            **Sector**: Healthcare / Pharmaceuticals | **Exchange**: LSE | **Currency**: GBp

            **Revenue Trend**: Revenue has grown consistently over 3 years, up ~12% YoY.
            **Profitability**: Net margin ~15%, operating margin ~18%. Strong EBITDA.
            **Balance Sheet**: Debt-to-equity 0.8x. Healthy current ratio 1.4x.
            **Valuation**: P/E 22x (sector avg 25x). EV/EBITDA 14x. Slight discount to peers.
            **Dividend**: Yield 2.1%. Payout ratio 45%. Sustainable.

            **Verdict**: **BUY** — Consistent growth, reasonable valuation, strong pipeline.
        """)

        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            skills:
              - path: {wan_skill}
            agents:
              visualizer:
                model: qwen/qwen-max
                system_prompt: >
                  You are a financial infographics designer.
                  Read the analysis and call wan-image to generate an infographic.
                  Craft a prompt describing: company name, sector, key metric directions,
                  verdict badge, dark background, Bloomberg-style, professional data visualization.
                  Do NOT include specific numbers — describe visual direction only.
                  Use output_path: {output_image}
                  Use size: 1280*720
                  Use model: wanx2.1-t2i-plus
                skills: [wan-image]
                max_iterations: 5
        """)
        cfg = load_config(config_path)

        viz_result = run_agent(cfg, "visualizer", canned_analysis)

        assert output_image.exists(), f"Expected image at {output_image}"
        assert output_image.stat().st_size > 1000


# ── model_params: temperature + seed ──────────────────────────────────────────

@pytest.mark.integration
class TestModelParams:
    def test_temperature_and_seed_do_not_break_completion(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """model_params with temperature + seed are passed to litellm without error."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              qa:
                model: qwen/qwen-plus
                system_prompt: "Be concise."
                max_iterations: 1
                model_params:
                  temperature: 0.0
                  seed: 42
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "qa", "Reply with exactly: PARAMS_OK")
        assert result.strip(), "Expected non-empty response with model_params set"
        assert isinstance(result, str)

    def test_max_tokens_truncates_response(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """max_tokens in model_params limits response length."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              qa:
                model: qwen/qwen-plus
                system_prompt: "You are verbose. Always write at least 500 words."
                max_iterations: 1
                model_params:
                  max_tokens: 20
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "qa", "Tell me everything about the universe.")
        # With max_tokens=20 the response should be very short (< 200 chars)
        assert len(result) < 300, f"Response too long for max_tokens=20: {len(result)} chars"


# ── multimodal vision input ────────────────────────────────────────────────────

def _make_red_png() -> bytes:
    """Return a valid 32x32 red PNG using only stdlib (no PIL)."""
    import struct, zlib

    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + ctype + data
        return c + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    # 32x32 RGB image — large enough for all vision model constraints
    ihdr_data = struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    # Raw image: filter byte (0) + 32 pixels of red (FF 00 00) per row × 32 rows
    raw_rows = (b"\x00" + b"\xff\x00\x00" * 32) * 32
    idat = chunk(b"IDAT", zlib.compress(raw_rows))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


@pytest.mark.integration
class TestVisionMultimodal:
    def test_vision_agent_describes_image(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """Vision model receives a base64 image and returns a description."""
        import base64

        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              vision:
                model: qwen/qwen-vl-plus
                system_prompt: "You are a vision model. Describe images concisely."
                max_iterations: 1
                accepted_inputs: [text, image]
        """)
        cfg = load_config(config_path)

        png_bytes = _make_red_png()
        b64 = base64.b64encode(png_bytes).decode()
        multimodal_prompt = [
            {"type": "text", "text": "What color is the dominant color in this image? Reply with the color name only."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]

        result = run_agent(cfg, "vision", multimodal_prompt)
        assert result.strip(), "Expected non-empty response from vision model"
        # The model should mention red in its response
        assert "red" in result.lower(), f"Expected 'red' in response, got: {result!r}"

    def test_call_agent_passes_image_to_sub_agent(
        self, tmp_path: Path, dashscope_api_key: str
    ) -> None:
        """Orchestrator passes an image file via call_agent attachments to a vision sub-agent."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              qwen:
                api_key: {dashscope_api_key}
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            agents:
              orchestrator:
                model: qwen/qwen-plus
                system_prompt: >
                  You are an orchestrator. When given an image file path,
                  call the vision sub-agent with that path in attachments
                  and report back what color the image is.
                max_iterations: 3
                agents: [vision]
              vision:
                model: qwen/qwen-vl-plus
                system_prompt: >
                  You are a vision expert. Describe the dominant color of any image
                  you receive. Reply with the color name only.
                max_iterations: 1
                accepted_inputs: [text, image]
        """)

        # Write the test image to a local path
        img_path = tmp_path / "red_square.png"
        img_path.write_bytes(_make_red_png())

        cfg = load_config(config_path)
        result = run_agent(
            cfg,
            "orchestrator",
            f"The image is at {img_path}. Use the vision agent with attachments to determine its color.",
        )
        assert result.strip(), "Expected non-empty orchestrator response"
        assert "red" in result.lower(), f"Expected 'red' mentioned in final response, got: {result!r}"
