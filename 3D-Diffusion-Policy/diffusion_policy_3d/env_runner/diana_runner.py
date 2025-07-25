from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.policy.base_policy import BasePolicy
from typing import Dict

class DianaRunner(BaseRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
