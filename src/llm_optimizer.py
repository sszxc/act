"""LLM-in-the-loop numerical optimizer (--method llm): prompt template loading/rendering,
optional VLM visual feedback / reward_predictor scoring, and the run_llm main loop."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.logging_utils import _save_progress_checkpoint
from src.optimizers import _handle_optimizer_interrupt


def _load_num_optim_prompt_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r'system_prompt\s*=\s*"""(.*?)"""', text, flags=re.S)
    if m is None:
        # Allow "raw prompt" files (plain text) too.
        # This enables alternative prompt templates that are not wrapped in a python variable.
        return text.strip()
    return m.group(1).strip()


def _prompt_template_is_vlm(path: Path) -> bool:
    """True if the prompt template file declares `IS_VLM = True` (see
    prompts/num_optim_Pratyush_vlm.py), i.e. it expects a visual-feedback image attached
    to every LLM call from iteration 1 onward, in addition to the text history."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*IS_VLM\s*=\s*True\s*$", text, flags=re.M))


def _prompt_template_uses_reward_predictor(path: Path) -> bool:
    """True if the prompt template file declares `IS_REWARD_PREDICTOR = True` (see
    prompts/num_optim_Pratyush_reward_predictor.py), i.e. it expects the local
    reward_predictor's text feedback (over its IPC bridge) on every LLM call from
    iteration 1 onward, in addition to the text history."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*IS_REWARD_PREDICTOR\s*=\s*True\s*$", text, flags=re.M))


def _format_history_text_for_prompt(history: list[dict], rank: int) -> str:
    """Format past evaluations for prompts/num_optim_Pratyush.py (maximize episode return)."""
    if not history:
        return "No previous samples yet."
    lines = []
    for rec in history:
        params = rec["params"]
        r = float(rec["reward"])
        params_str = ", ".join([f"params[{i}]: {params[i]:.2f}" for i in range(rank)])
        lines.append(f"iter {rec['iter']}: {params_str}, R(params): {r:.6f}")
    return "\n".join(lines)


def _render_num_optim_prompt(
    template: str,
    *,
    rank: int,
    optimum_reward: float,
    step_size: float,
    episode_num: int,
    total_episodes: int,
    history_text: str,
    reward_predictor_text: str = "",
) -> str:
    """Fill placeholders for prompts/num_optim_Pratyush.py-style templates (no Jinja).
    `reward_predictor_text` fills {{ reward_predictor_text }}, used only by
    prompts/num_optim_Pratyush_reward_predictor.py-style templates (no-op otherwise)."""
    rendered = template
    rendered = rendered.replace("{{ rank - 1 }}", str(rank - 1))
    rendered = rendered.replace("{{ rank }}", str(rank))

    rendered = re.sub(
        r"\{\{\s*optimum_reward:\.1f\s*\}\}",
        f"{float(optimum_reward):.1f}",
        rendered,
    )
    rendered = re.sub(r"\{\{\s*optimum_reward\s*\}\}", f"{float(optimum_reward)}", rendered)

    rendered = re.sub(r"\{\{\s*step_size\s*\}\}", str(float(step_size)), rendered)
    rendered = re.sub(r"\{\{\s*episode_num\s*\}\}", str(int(episode_num)), rendered)
    rendered = re.sub(r"\{\{\s*total_episodes\s*\}\}", str(int(total_episodes)), rendered)
    rendered = re.sub(r"\{\{\s*history_text\s*\}\}", history_text, rendered)
    rendered = re.sub(r"\{\{\s*reward_predictor_text\s*\}\}", reward_predictor_text, rendered)

    rendered = re.sub(r"<start_of_turn>.*$", "", rendered, flags=re.M).strip()
    return rendered


def _parse_llm_params_response(text: str, rank: int) -> np.ndarray | None:
    if not text:
        return None
    m = re.search(r"<param\b[^>]*>(.*?)</param\s*>", text, flags=re.S | re.I)
    scope = m.group(1) if m else text
    matches = re.findall(r"params\[(\d+)\]\s*:\s*([-+]?\d*\.?\d+)", scope)
    if matches:
        vals = np.full(rank, np.nan, dtype=np.float64)
        for idx_str, val_str in matches:
            idx = int(idx_str)
            if 0 <= idx < rank and np.isnan(vals[idx]):
                vals[idx] = float(val_str)
        if np.all(np.isfinite(vals)):
            return vals
    raw_nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", scope)
    if len(raw_nums) >= rank:
        return np.asarray([float(x) for x in raw_nums[:rank]], dtype=np.float64)
    return None


def _clip_quantize_params(x: np.ndarray, low: float = -6.0, high: float = 6.0, decimals: int = 1) -> np.ndarray:
    return np.round(np.clip(np.asarray(x, dtype=np.float64), low, high), decimals)


def _sample_unseen_params(rng: np.random.Generator, rank: int, seen: set[tuple[float, ...]]) -> np.ndarray:
    for _ in range(4096):
        cand = rng.integers(-600, 601, size=rank).astype(np.float64) / 100.0
        key = tuple(float(v) for v in cand.tolist())
        if key not in seen:
            return cand
    # Extremely unlikely fallback.
    cand = rng.uniform(-6.0, 6.0, size=rank)
    return _clip_quantize_params(cand)


def _init_openai_compatible_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
        import httpx
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Please install openai package for --method llm: `pip install openai`") from e
    # Library default read timeout is 600s; a stuck-but-connected SOL request can silently
    # block that long per attempt. 300s still gives "thinking" models room to generate while
    # failing a truly hung request in reasonable time. connect stays short (5s) so a fully-down
    # endpoint fails fast.
    timeout = httpx.Timeout(300.0, connect=5.0)
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _maybe_load_dotenv():
    # Optional: allow local `.env` without forcing a dependency.
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(override=False)


def _get_env(name: str) -> str | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def _build_vlm_feedback_image(image_dicts: list[dict], num_frames: int = 5) -> np.ndarray:
    """Sample `num_frames` evenly-spaced timesteps from one rollout's captured per-step camera
    images (same list produced by rollout_batch_episode_returns(..., capture_frames=True)),
    label each with its frame index, and stack them top-to-bottom into a single RGB uint8
    composite image. Multiple cameras appear side-by-side within each row, matching
    visualize_episodes.save_videos' frame layout.
    """
    import cv2  # local import: only needed for the VLM prompt path

    total = len(image_dicts)
    if total == 0:
        raise ValueError("No frames captured for VLM feedback")
    n = max(1, min(num_frames, total))
    indices = [total - 1] if n == 1 else sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})

    cam_names = list(image_dicts[0].keys())
    rows = []
    for idx in indices:
        cams = [np.asarray(image_dicts[idx][cam]) for cam in cam_names]
        row = np.concatenate(cams, axis=1).copy()  # side-by-side cameras, HxWxC uint8 RGB
        label = f"frame {idx}"
        cv2.putText(row, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(row, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)

    max_w = max(r.shape[1] for r in rows)
    if any(r.shape[1] != max_w for r in rows):
        rows = [
            r if r.shape[1] == max_w else np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)))
            for r in rows
        ]
    return np.concatenate(rows, axis=0)


def _encode_image_array_to_data_uri(img: np.ndarray, quality: int = 90) -> str:
    """RGB uint8 HxWxC array -> `data:image/jpeg;base64,...` string usable as an image_url."""
    import cv2  # local import: only needed for the VLM prompt path

    bgr = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR for cv2.imencode
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode VLM feedback image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _score_video_with_reward_predictor(
    video_paths: list[str],
    *,
    instruction: str,
    camera_labels: list[str],
    socket_path: str,
    timeout: float,
    output_dir: Path,
) -> dict:
    """Score one rollout's synchronized per-camera videos via the local reward_predictor's
    IPC bridge: whole-video scoring (runner="main_single", not the process-reward-curve
    runner), subgoals auto-generated from `instruction` on every call (no `subgoals=`
    passed). Requires the reward_predictor repo to already be on sys.path (see main())
    and its `python -m ipc.server` running with OPENAI_API_KEY sourced (for subgoal
    generation). Returns {"progress_reward", "success_score", "components"}."""
    from ipc.client import score_trajectory  # local import: only needed for this path

    if len(video_paths) != len(camera_labels):
        raise ValueError(
            f"got {len(video_paths)} videos but {len(camera_labels)} camera_labels"
        )
    if not 1 <= len(video_paths) <= 5:
        raise ValueError(f"reward_predictor supports 1-5 synchronized videos, got {len(video_paths)}")
    video_kwargs = {f"video{i}": p for i, p in enumerate(video_paths, start=1)}

    return score_trajectory(
        instruction,
        **video_kwargs,
        runner="main_single",
        camera_labels=camera_labels,
        socket_path=socket_path,
        timeout=timeout,
        output_dir=str(output_dir),
    )


def _format_reward_predictor_feedback_text(result: dict) -> str:
    progress = float(result.get("progress_reward", float("nan")))
    success = float(result.get("success_score", float("nan")))
    lines = [f"progress_reward: {progress:.3f}, success_score: {success:.3f}"]
    for name, prob in (result.get("components") or {}).items():
        lines.append(f"  {name}: {float(prob):.3f}")
    return "\n".join(lines)


def _call_llm_next_params(
    client,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int | None = None,
    image_data_uri: str | None = None,
) -> tuple[str, str]:
    """Returns (content, reasoning) — reasoning is the model's separate thinking-channel
    text when the backend exposes one (see below), else ""."""
    if image_data_uri:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        content = prompt
    kwargs = {}
    if max_tokens is not None:
        # "Thinking" models spend generation budget on a <think> block before the final
        # answer; too small a budget (or an unset one, on some backends) truncates mid-thought
        # and message.content comes back empty (finish_reason="length") even though the API
        # call itself succeeded. An explicit, generous budget makes that failure mode rare and,
        # when it still happens, distinguishable via finish_reason rather than silently empty.
        kwargs["max_tokens"] = int(max_tokens)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=float(temperature),
        **kwargs,
    )
    message = resp.choices[0].message
    content_resp = message.content
    # "Thinking" models served over OpenAI-compatible backends (vLLM/SGLang reasoning
    # parsers, etc.) typically split their chain-of-thought out of `message.content` and
    # into a separate field — most commonly `reasoning_content`, occasionally `reasoning`.
    # Without capturing it, the logged response is just the bare <param> block with no way
    # to audit whether the model actually engaged with the required <think> analysis (state/
    # visual/trend/sensitivity/mode) or looked at the attached image.
    reasoning_resp = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    return (content_resp if content_resp is not None else ""), (reasoning_resp or "")


def run_llm(
    fitness_fn,
    x0: np.ndarray,
    *,
    maxiter: int,
    seed: int,
    log_path: Path,
    llm_model: str,
    llm_temperature: float,
    llm_max_retries: int,
    llm_max_tokens: int | None,
    llm_retry_temperature_bump: float,
    llm_history_window: int,
    llm_step_size_hint: float,
    llm_optimum_hint: float,
    prompt_template_path: Path,
    vlm_fitness_fn=None,
    vlm_num_frames: int = 5,
    rp_fitness_fn=None,
    rp_instruction: str | None = None,
    rp_camera_labels: list[str] | None = None,
    rp_socket_path: str = "/tmp/reward_predictor_ipc.sock",
    rp_timeout: float = 3600.0,
    rp_output_dir: Path | None = None,
):
    """
    vlm_fitness_fn: optional Callable[[np.ndarray], tuple[float, list[dict]]] — same role as
    fitness_fn but also returns the evaluated candidate's per-timestep captured images (see
    rollout_batch_episode_returns(..., capture_frames=True)). When given (i.e. the prompt
    template sets IS_VLM = True, see prompts/num_optim_Pratyush_vlm.py), it replaces fitness_fn
    for every evaluation, and a composite image built from vlm_num_frames evenly-spaced frames
    of the most recently evaluated rollout is attached to every LLM call from iteration 1 onward
    (the text history passed in the prompt still spans all iterations; only the image is
    last-round-only).

    rp_fitness_fn: optional Callable[[np.ndarray], tuple[float, list[str]]] — mutually exclusive
    with vlm_fitness_fn. When given (i.e. the prompt template sets IS_REWARD_PREDICTOR = True, see
    prompts/num_optim_Pratyush_reward_predictor.py), it replaces fitness_fn for every rollout, but
    NOT the reward: its own return value (episode_return) is only kept as diagnostic history
    metadata, never used as R(params). Instead, the local reward_predictor (over its IPC bridge,
    rp_socket_path) scores the returned per-camera videos (one per rp_camera_labels entry, kept
    separate — never merged into one side-by-side frame) against auto-generated subgoals for
    rp_instruction, and R(params) becomes that call's progress_reward. A text breakdown of the
    score (success_score + per-subgoal probabilities) is attached to every LLM call from
    iteration 1 onward (last-round-only, same cadence as vlm_fitness_fn's image).
    """
    rank = int(np.asarray(x0).size)
    out_dir = log_path.parent
    rng = np.random.default_rng(seed)
    _maybe_load_dotenv()
    base_url = _get_env("OPENAI_BASE_URL") or "https://openai.rc.asu.edu/v1"
    api_key = _get_env("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your shell (recommended) or in a local `.env` file.\n"
            "Example:\n"
            "  export OPENAI_API_KEY='...'\n"
            "Optional:\n"
            "  export OPENAI_BASE_URL='https://openai.rc.asu.edu/v1'"
        )
    client = _init_openai_compatible_client(
        base_url=base_url,
        api_key=api_key,
    )
    template = _load_num_optim_prompt_template(prompt_template_path)
    prompt_log_dir = log_path.parent / "llm_prompt_logs"
    prompt_log_dir.mkdir(parents=True, exist_ok=True)
    if prompt_template_path.is_file():
        shutil.copy2(prompt_template_path, prompt_log_dir / prompt_template_path.name)

    seen: set[tuple[float, ...]] = set()
    history: list[dict] = []
    history_best = []
    history_iter_reward = []
    best_so_far = -np.inf
    best_x = _clip_quantize_params(x0)
    use_vlm = vlm_fitness_fn is not None
    last_vlm_image_uri: str | None = None  # visual feedback from the previous iteration's rollout only
    use_reward_predictor = rp_fitness_fn is not None
    last_rp_feedback_text = "No reward-predictor feedback yet (first iteration)."

    t_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write(
            f"# LLM dim={rank} maxiter={maxiter} model={llm_model} temp={llm_temperature} "
            f"max_retries={llm_max_retries} max_tokens={llm_max_tokens} "
            f"retry_temperature_bump={llm_retry_temperature_bump} start_time={datetime.now().isoformat()}\n"
        )

    interrupted_exc: BaseException | None = None
    it = -1
    try:
        for it in range(maxiter):
            t0 = time.perf_counter()
            raw_response = ""
            source = "seed"
            extra_history_fields: dict = {}
            if it == 0:
                cand = _clip_quantize_params(x0)
            else:
                prompt_hist = history[-llm_history_window:] if llm_history_window > 0 else history
                history_text = _format_history_text_for_prompt(prompt_hist, rank)
                prompt = _render_num_optim_prompt(
                    template,
                    rank=rank,
                    optimum_reward=float(llm_optimum_hint),
                    step_size=llm_step_size_hint,
                    episode_num=it + 1,
                    total_episodes=maxiter,
                    history_text=history_text,
                    reward_predictor_text=last_rp_feedback_text,
                )
                (prompt_log_dir / f"rendered_prompt_iter_{it:04d}.txt").write_text(prompt, encoding="utf-8")
                cand = None
                n_tries = max(1, int(llm_max_retries))
                every_call_raised = True
                saw_parseable_vector = False
                attempt_responses: list[str] = []
                for retry in range(n_tries):
                    if retry > 0:
                        # Backoff before re-hitting SOL after any failed attempt (error, parse
                        # failure, or duplicate) — the openai SDK already backs off *within* one
                        # call's internal retries, but nothing previously paced *these* app-level
                        # retries, so a down/rate-limited endpoint got hit back-to-back.
                        # 10s, 20s, 40s, 80s, 160s, then plateaus (cap = 5th doubling).
                        backoff_s = min(10.0 * 2.0 ** (retry - 1), 160.0)
                        print(f"[LLM] iter {it}: previous attempt failed, backing off {backoff_s:.0f}s before retry {retry}")
                        time.sleep(backoff_s)
                    # Retries reuse the exact same prompt, so at low temperature the model tends
                    # to just repeat its previous (already-seen) answer, or the same truncated-
                    # empty response, verbatim — bumping temperature per retry gives it a real
                    # chance to land somewhere different instead of burning the retry budget on
                    # a near-deterministic repeat.
                    retry_temperature = min(2.0, llm_temperature + retry * llm_retry_temperature_bump)
                    try:
                        print(
                            f"[LLM] iter {it}/{maxiter-1} requesting next params from model={llm_model} "
                            f"(retry {retry}/{n_tries - 1}, temp={retry_temperature:.2f}) — waiting for optimizer response"
                        )
                        raw_response, reasoning_response = _call_llm_next_params(
                            client,
                            model=llm_model,
                            prompt=prompt,
                            temperature=retry_temperature,
                            max_tokens=llm_max_tokens,
                            image_data_uri=last_vlm_image_uri if use_vlm else None,
                        )
                        attempt_log = f"--- retry {retry} (temp={retry_temperature:.2f}) ---\n"
                        if reasoning_response:
                            # Logged for audit only — never fed back into history_text or parsed
                            # for <param>, so it can't change downstream behavior.
                            attempt_log += f"--- reasoning ---\n{reasoning_response}\n--- content ---\n"
                        attempt_log += raw_response
                        attempt_responses.append(attempt_log)
                        every_call_raised = False
                        parsed = _parse_llm_params_response(raw_response, rank)
                        if parsed is None:
                            continue
                        saw_parseable_vector = True
                        parsed = _clip_quantize_params(parsed)
                        key = tuple(float(v) for v in parsed.tolist())
                        if key in seen:
                            continue
                        cand = parsed
                        source = f"llm_success_after_retry_{retry}" if retry > 0 else "llm_success"
                        break
                    except Exception as e:
                        raw_response = f"__llm_error__: {e}"
                        attempt_responses.append(f"--- retry {retry} (temp={retry_temperature:.2f}) ---\n{raw_response}")
                (prompt_log_dir / f"rendered_response_iter_{it:04d}.txt").write_text(
                    "\n\n".join(attempt_responses), encoding="utf-8"
                )
                if cand is None:
                    cand = _sample_unseen_params(rng, rank, seen)
                    # Distinguish why the LLM path failed (logged in source / optional fallback_reason).
                    if saw_parseable_vector:
                        source = "duplicate_after_retries"
                    elif every_call_raised:
                        source = "api_error"
                    else:
                        source = "parse_failure"

            cand = _clip_quantize_params(cand)
            key = tuple(float(v) for v in cand.tolist())
            if key in seen:
                cand = _sample_unseen_params(rng, rank, seen)
                key = tuple(float(v) for v in cand.tolist())
                source = f"{source}_dedup"

            if use_vlm:
                reward, frames = vlm_fitness_fn(cand)
                reward = float(reward)
                try:
                    composite = _build_vlm_feedback_image(frames, num_frames=vlm_num_frames)
                    last_vlm_image_uri = _encode_image_array_to_data_uri(composite)
                    # Save every round's composite unconditionally (not just when it ends up
                    # attached to a later prompt) so round 0 and the final round are covered too.
                    (prompt_log_dir / f"vlm_feedback_round_{it:04d}.jpg").write_bytes(
                        base64.b64decode(last_vlm_image_uri.split(",", 1)[1])
                    )
                except Exception as e:
                    print(f"[LLM] iter {it}: warning: failed to build VLM feedback image: {e}")
                    last_vlm_image_uri = None
            elif use_reward_predictor:
                # R(params) is now the reward_predictor's progress_reward, not episode_return — a
                # rollout still happens (to produce the video) but its own env reward is only kept
                # as diagnostic history metadata below, never as the optimization objective. A
                # scoring failure here is fatal to this round (there is no fallback reward to fall
                # back on) and is intentionally left to propagate to the run's outer except-block,
                # same as any other fitness-evaluation failure.
                rollout_return, video_paths = rp_fitness_fn(cand)
                rp_result = _score_video_with_reward_predictor(
                    video_paths,
                    instruction=rp_instruction,
                    camera_labels=rp_camera_labels,
                    socket_path=rp_socket_path,
                    timeout=rp_timeout,
                    output_dir=rp_output_dir / f"round_{it:04d}",
                )
                reward = float(rp_result["progress_reward"])
                extra_history_fields["rollout_episode_return"] = float(rollout_return)
                extra_history_fields["rp_success_score"] = float(rp_result.get("success_score", float("nan")))
                last_rp_feedback_text = _format_reward_predictor_feedback_text(rp_result)
                (prompt_log_dir / f"rp_feedback_round_{it:04d}.json").write_text(
                    json.dumps(rp_result, indent=2), encoding="utf-8"
                )
            else:
                reward = float(fitness_fn(cand))
            seen.add(key)

            if reward > best_so_far:
                best_so_far = reward
                best_x = cand.copy()

            history.append(
                {
                    "iter": it,
                    "params": [float(v) for v in cand.tolist()],
                    "reward": reward,
                    "best_so_far": float(best_so_far),
                    "source": source,
                    "raw_response": raw_response,
                    "elapsed_sec": round(time.perf_counter() - t_start, 3),
                    **extra_history_fields,
                }
            )
            history_best.append(float(best_so_far))
            history_iter_reward.append(float(reward))

            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(json.dumps(history[-1], ensure_ascii=False) + "\n")
            elapsed_now = time.perf_counter() - t_start
            _save_progress_checkpoint(
                out_dir,
                "llm_curves.npz",
                {"best_so_far": np.array(history_best), "iter_reward": np.array(history_iter_reward)},
                np.array(history_best),
                "best_so_far",
                np.array(history_iter_reward),
                "iter_reward",
                elapsed_now,
                ylabel="progress_reward" if use_reward_predictor else "episode_return",
            )
            dt = time.perf_counter() - t0
            reward_kind = "progress_reward" if use_reward_predictor else "reward"
            print(f"[LLM] iter {it}: {reward_kind}={reward:.4f} best_so_far={best_so_far:.4f} source={source} wall={dt:.2f}s")
    except (KeyboardInterrupt, Exception) as e:
        interrupted_exc = e
        _handle_optimizer_interrupt("LLM", e, it, len(history_best))

    elapsed_sec = time.perf_counter() - t_start
    return best_x, np.asarray(history_best), np.asarray(history_iter_reward), interrupted_exc, elapsed_sec
