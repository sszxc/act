#!/usr/bin/env python
"""Quick CLI to smoke-test an OpenAI-compatible LLM/VLM API: pick a model,
pass a prompt (optionally with images), see the response and the latency.

Usage:
    python test_llm.py "Hello!"
    python test_llm.py --model qwen3-30b-a3b-thinking-2507 "Explain FiLM conditioning"
    echo "Hello!" | python test_llm.py --model gemma4-31b-it
    python test_llm.py --model qwen3-vl-32b-thinking --image assets/example.jpg "What is in this image?"
    python test_llm.py --model qwen3-vl-32b-thinking -i img1.png -i img2.png "Compare these two images"
    python test_llm.py --model qwen3-vl-32b-thinking --video clip.mp4 "What happens in this video?"
    python test_llm.py --list-models

Images can be a local file path (base64-encoded automatically) or an
http(s) URL. Native video inputs (video_url) are rejected by the current
backend ("At most 0 video(s) may be provided"), so --video instead samples
evenly-spaced frames from a local video file with OpenCV and sends them as
a sequence of images, the same way -i/--image does. Requires OPENAI_API_KEY
(and optionally OPENAI_BASE_URL) in the environment or a local `.env` file
(see .env.example).
"""
import argparse
import base64
import mimetypes
import os
import sys
import time

from openai import OpenAI

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(override=False)

# Models known to be available on the configured endpoint. Add/remove as needed;
# any model name string still works even if it's not in this list.
KNOWN_MODELS = [
    "llama4-maverick-17b",
    "gemma4-31b-it",
    "qwen3-30b-a3b-thinking-2507",
    "qwen35-27b",
    "qwen36-27b",
    "qwen38-27b",
    "qwen3-vl-32b-thinking",
]
# Models that accept image inputs (used only for a heads-up warning below).
VISION_MODELS = {
    "qwen3-vl-32b-thinking",
}
DEFAULT_MODEL = KNOWN_MODELS[0]


def encode_image(spec: str) -> str:
    """Turn a local path or URL into an `image_url` value the API accepts."""
    if spec.startswith(("http://", "https://", "data:")):
        return spec
    mime, _ = mimetypes.guess_type(spec)
    mime = mime or "image/png"
    with open(spec, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_video_frames(path: str, num_frames: int) -> list[str]:
    """Sample `num_frames` evenly-spaced frames from a local video file and
    return them as base64-encoded JPEG data URIs (usable as `image_url` values).
    """
    import cv2  # local import: only needed when --video is used

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine frame count for video: {path}")

    n = max(1, min(num_frames, total))
    if n == 1:
        indices = [total // 2]
    else:
        indices = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        frames.append(f"data:image/jpeg;base64,{b64}")
    cap.release()

    if not frames:
        raise RuntimeError(f"Failed to extract any frames from video: {path}")
    return frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a prompt to an OpenAI-compatible LLM API and report latency."
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text. If omitted, read from stdin.")
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})."
    )
    parser.add_argument("-s", "--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "-i",
        "--image",
        action="append",
        default=None,
        metavar="PATH_OR_URL",
        help="Image to attach (local file path or URL). May be passed multiple times.",
    )
    parser.add_argument(
        "--video",
        default=None,
        metavar="PATH",
        help=(
            "Local video file to attach. Native video input isn't supported by the "
            "backend, so this samples evenly-spaced frames (see --video-frames) and "
            "sends them as images instead."
        ),
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Number of frames to sample from --video (default: 4). Note: the "
            "current backend caps requests at 4 images total (image + video "
            "frames combined), so higher values will likely 400."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature.")
    parser.add_argument(
        "--list-models", action="store_true", help="Print known model names and exit."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("Known models (edit KNOWN_MODELS in test_llm.py to add more):")
        for m in KNOWN_MODELS:
            tag = " (vision)" if m in VISION_MODELS else ""
            print(f"  - {m}{tag}")
        return

    prompt = args.prompt
    if prompt is None:
        if sys.stdin.isatty():
            print("Enter prompt (Ctrl-D to send):", file=sys.stderr)
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt given.", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your shell or a local `.env` file.\n"
            "Example: export OPENAI_API_KEY='...'\n"
            "Optional: export OPENAI_BASE_URL='https://openai.rc.asu.edu/v1'"
        )

    client = OpenAI(base_url=base_url, api_key=api_key)

    if (args.image or args.video) and args.model not in VISION_MODELS:
        print(
            f"Warning: '{args.model}' isn't in the known VISION_MODELS list; "
            "the API may reject the image/video input.",
            file=sys.stderr,
        )

    video_frames = []
    if args.video:
        print(f"Extracting {args.video_frames} frame(s) from {args.video}...", file=sys.stderr)
        video_frames = extract_video_frames(args.video, args.video_frames)

    total_images = len(args.image or []) + len(video_frames)
    if total_images > 4:
        print(
            f"Warning: sending {total_images} image(s) total; the current backend "
            "caps requests at 4 images and will likely reject this.",
            file=sys.stderr,
        )

    if args.image or video_frames:
        user_content = [{"type": "text", "text": prompt}]
        for spec in args.image or []:
            user_content.append({"type": "image_url", "image_url": {"url": encode_image(spec)}})
        for frame_uri in video_frames:
            user_content.append({"type": "image_url", "image_url": {"url": frame_uri}})
        user_message = {"role": "user", "content": user_content}
    else:
        user_message = {"role": "user", "content": prompt}

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append(user_message)

    kwargs = {"model": args.model, "messages": messages}
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    print(f"Model: {args.model}")
    print(f"Prompt: {prompt}")
    if args.image:
        print(f"Images: {', '.join(args.image)}")
    if video_frames:
        print(f"Video: {args.video} ({len(video_frames)} frame(s) sampled)")
    print("Waiting for response...")

    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0

    content = response.choices[0].message.content
    usage = getattr(response, "usage", None)

    print("\n--- Response ---")
    print(content)
    print("----------------")
    print(f"Latency: {elapsed:.3f}s")
    if usage is not None:
        print(
            f"Tokens: prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}"
        )


if __name__ == "__main__":
    main()
