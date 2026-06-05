#!/usr/bin/env python3
"""Sample records with image prompts and materialize local input images via Qwen-Image-2.0-Pro."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError


DEFAULT_INPUT = Path('/nfsdata4/zhuhairui/verl/examples/qwen3vl_multiturn/zhu/qwen_stage12_full_2000.json')
DEFAULT_OUTPUT = Path('/nfsdata4/zhuhairui/verl/examples/qwen3vl_multiturn/zhu/qwen_stage12_sampled20_with_local_images.json')
DEFAULT_IMAGE_DIR = Path('/nfsdata4/zhuhairui/RL_image')
DEFAULT_ENDPOINT = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
DEFAULT_MODEL = 'qwen-image-2.0'
DEFAULT_NEGATIVE_PROMPT = (
    'Low resolution, low quality, distorted anatomy, malformed objects, broken text, '
    'oversaturated colors, AI-looking artifacts, chaotic composition, blurry details.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--image-dir', type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--sample-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260409)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    parser.add_argument('--size', default='512*512')
    parser.add_argument('--prompt-extend', action='store_true', default=True)
    parser.add_argument('--no-prompt-extend', dest='prompt_extend', action='store_false')
    parser.add_argument('--watermark', action='store_true', default=False)
    parser.add_argument('--negative-prompt', default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--sleep-between-requests', type=float, default=6.0)
    parser.add_argument('--flush-every', type=int, default=20)
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--include-prompt-only', action='store_true', default=False)
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))


def select_samples(samples: list[dict[str, Any]], sample_size: int, seed: int, include_prompt_only: bool) -> list[dict[str, Any]]:
    if include_prompt_only:
        candidates = samples
    else:
        candidates = [sample for sample in samples if sample.get('result', {}).get('image_prompt')]
    if sample_size > len(candidates):
        raise ValueError(f'Requested {sample_size} samples but only {len(candidates)} candidates are available.')
    rng = random.Random(seed)
    selected = rng.sample(candidates, sample_size)
    selected.sort(key=lambda item: item['sample_id'])
    return selected


def build_payload(prompt: str, model: str, size: str, negative_prompt: str, prompt_extend: bool, watermark: bool) -> dict[str, Any]:
    return {
        'model': model,
        'input': {
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'text': prompt},
                    ],
                }
            ]
        },
        'parameters': {
            'negative_prompt': negative_prompt,
            'prompt_extend': prompt_extend,
            'watermark': watermark,
            'size': size,
        },
    }


def call_qwen_image(endpoint: str, api_key: str, payload: dict[str, Any], timeout: int, max_retries: int) -> dict[str, Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if data.get('output', {}).get('choices'):
                return data
            raise RuntimeError(f'Unexpected response payload: {json.dumps(data, ensure_ascii=False)[:1000]}')
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                wait_seconds = min(5 * attempt, 15)
                if isinstance(exc, HTTPError) and exc.response is not None and exc.response.status_code == 429:
                    wait_seconds = min(30 * attempt, 90)
                time.sleep(wait_seconds)
    raise RuntimeError(f'Qwen image generation failed after {max_retries} attempts: {last_error}')


def extract_image_urls(response_json: dict[str, Any]) -> list[str]:
    content = response_json['output']['choices'][0]['message']['content']
    return [item['image'] for item in content if 'image' in item]


def download_image(url: str, destination: Path, timeout: int) -> None:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)


def render_progress(
    sample_done: int,
    sample_total: int,
    image_done: int,
    image_total: int,
    reused_images: int,
    start_time: float,
) -> str:
    width = 32
    ratio = sample_done / sample_total if sample_total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_time, 1e-6)
    rate = sample_done / elapsed
    remaining = max(sample_total - sample_done, 0)
    eta = remaining / rate if rate > 0 else 0
    return (
        f"\r[{bar}] {sample_done}/{sample_total} samples "
        f"({ratio * 100:5.1f}%) | images {image_done}/{image_total} "
        f"| reused {reused_images} | elapsed {elapsed / 60:5.1f}m | eta {eta / 60:5.1f}m"
    )


def print_progress(
    sample_done: int,
    sample_total: int,
    image_done: int,
    image_total: int,
    reused_images: int,
    start_time: float,
    disabled: bool,
) -> None:
    if disabled:
        return
    sys.stderr.write(render_progress(sample_done, sample_total, image_done, image_total, reused_images, start_time))
    sys.stderr.flush()


def materialize_sample(
    sample: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> tuple[dict[str, Any], int, int]:
    sample_copy = json.loads(json.dumps(sample, ensure_ascii=False))
    prompts = sample_copy.get('result', {}).get('image_prompt', [])
    asset_roles = sample_copy.get('input_spec', {}).get('asset_roles', [])
    generated_images: list[dict[str, Any]] = []
    generated_count = 0
    reused_count = 0

    for idx, prompt in enumerate(prompts):
        slot = asset_roles[idx]['slot'] if idx < len(asset_roles) else f'img_{idx + 1}'
        filename = f"{sample_copy['sample_id']}_{slot}.png"
        image_path = args.image_dir / filename
        if not image_path.exists() or args.overwrite:
            payload = build_payload(
                prompt=prompt,
                model=args.model,
                size=args.size,
                negative_prompt=args.negative_prompt,
                prompt_extend=args.prompt_extend,
                watermark=args.watermark,
            )
            response_json = call_qwen_image(args.endpoint, api_key, payload, args.timeout, args.max_retries)
            image_urls = extract_image_urls(response_json)
            if not image_urls:
                raise RuntimeError(f'No image URL returned for {sample_copy["sample_id"]} {slot}')
            download_image(image_urls[0], image_path, args.timeout)
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)
            generation_meta = {
                'model': args.model,
                'endpoint': args.endpoint,
                'size': args.size,
                'source_url': image_urls[0],
            }
            generated_count += 1
        else:
            generation_meta = {
                'model': args.model,
                'endpoint': args.endpoint,
                'size': args.size,
                'source_url': None,
                'reused_existing_file': True,
            }
            reused_count += 1

        generated_images.append(
            {
                'slot': slot,
                'image': str(image_path.resolve()),
                'image_prompt': prompt,
                'asset_role': asset_roles[idx] if idx < len(asset_roles) else None,
                'generation_meta': generation_meta,
            }
        )

    sample_copy['images'] = generated_images
    sample_copy['image_paths'] = [item['image'] for item in generated_images]
    return sample_copy, generated_count, reused_count


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        raise SystemExit('Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.')

    args.image_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.input)
    selected = select_samples(samples, args.sample_size, args.seed, args.include_prompt_only)
    total_images = sum(len(sample.get('result', {}).get('image_prompt', [])) for sample in selected)

    results = []
    generated_images = 0
    reused_images = 0
    start_time = time.time()
    print_progress(0, len(selected), 0, total_images, 0, start_time, args.no_progress)
    for sample_idx, sample in enumerate(selected, start=1):
        materialized, generated_count, reused_count = materialize_sample(sample, args, api_key)
        results.append(materialized)
        generated_images += generated_count
        reused_images += reused_count
        done_images = generated_images + reused_images
        print_progress(sample_idx, len(selected), done_images, total_images, reused_images, start_time, args.no_progress)
        if args.flush_every > 0 and sample_idx % args.flush_every == 0:
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.no_progress:
        sys.stderr.write('\n')
    print(f'saved={args.output}')
    print(f'sampled={len(results)}')
    print(f'images_total={total_images}')
    print(f'images_generated={generated_images}')
    print(f'images_reused={reused_images}')
    print(f'image_dir={args.image_dir.resolve()}')


if __name__ == '__main__':
    main()
