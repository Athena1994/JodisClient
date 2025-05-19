
import asyncio
import json
from pathlib import Path
import sys

from core.http_api.http_api import HttpAPI
from utils.config.attribute import Attribute
from utils.config.decorator import config
from utils.config.helper import load_config


@config
class Config:
    job_config: Path = Attribute('job-config', json_type=str, required=True)
    module_id: str = Attribute('module-id', default='')
    module_name: str = Attribute('module-name', default='')

    url: str = Attribute('url', default='http://localhost:5000')


async def main():
    if len(sys.argv) < 2:
        print("Usage: python create.py <job-config> <job-name>")
        return

    cfg_path = Path(sys.argv[1])
    try:
        cfg = load_config(cfg_path, Config)
        print(f"Loaded config: {cfg}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    api = HttpAPI(cfg.url)

    try:
        if cfg.module_id != '':
            module = api.modules.get_module(cfg.module_id)
            cfg.module_name = module.name
        elif cfg.module_name != '':
            module = api.modules.get_module_versions(cfg.module_name)[0]
            cfg.module_id = module.id
        else:
            print("No module id or name provided.")
            return
    except Exception as e:
        print(f"Error getting module: {e}")
        return

    name = sys.argv[2] if len(sys.argv) > 2 else f"{cfg.module_name}-job"

    job_config_path = Path(cfg_path.parent, cfg.job_config)
    try:
        with open(job_config_path, 'r') as f:
            job_config = json.load(f)
    except Exception as e:
        print(f"Error loading job config: {e}")
        return

    try:
        if not api.jobs.validate(cfg.module_id, job_config):
            print("Job config is not valid.")
            return
    except Exception as e:
        print(f"Error validating job config: {e}")
        return

    try:
        job = api.jobs.create(cfg.module_id, job_config, name)
        print(f"Job created successfully: {job}")
    except Exception as e:
        print(f"Error creating job: {e}")
        return


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
