from __future__ import annotations

import argparse
import asyncio

import yaml

from almox_vision.pipeline import AlmoxarifadoPipeline


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    pipeline = AlmoxarifadoPipeline(config)
    tasks = [pipeline.process_camera(url, cam_id) for cam_id, url in config["camera_urls"].items()]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
