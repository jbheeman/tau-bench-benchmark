# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list or config.model_provider == "local_hf", "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    attempt_suffix = f"_attempt-{config.attempt_id}" if config.attempt_id else ""
    temp_mode = (
        f"_tempU-{config.temperature_sampling_min}-{config.temperature_sampling_max}"
        if config.temperature_sampling_no_shift
        else ""
    )
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}{temp_mode}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}{attempt_suffix}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    # Determine task IDs once (shuffled if needed)
    idxs = list(range(len(env.tasks)))
    if config.shuffle:
        random.shuffle(idxs)
    
    if config.task_ids and len(config.task_ids) > 0:
        idxs = config.task_ids
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        idxs = idxs[config.start_index: end_index]
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )
    
    # Run each task num_trials times
    for idx in idxs:
        for i in range(config.num_trials):
            # Best-of-N trial setup: either sample perturbations (shift) OR sample temperature (no shift).
            chosen_sigma = 0.0
            if config.model_provider == "local_hf" and config.temperature_sampling_no_shift:
                # Ensure no perturbation shift is active, and let the agent sample random temperature instead.
                if hasattr(agent, "reset_perturbations"):
                    agent.reset_perturbations()
                if hasattr(agent, "set_temperature_sampling_no_shift"):
                    # First trial (i == 0) always uses greedy decoding
                    if i == 0:
                        agent.set_temperature_sampling_no_shift(
                            enabled=False,
                            t_min=config.temperature_sampling_min,
                            t_max=config.temperature_sampling_max,
                        )
                        print(f"Task {idx}, Trial {i+1}/{config.num_trials}: Greedy decoding (first trial, no perturbation shift)")
                    else:
                        agent.set_temperature_sampling_no_shift(
                            enabled=True,
                            t_min=config.temperature_sampling_min,
                            t_max=config.temperature_sampling_max,
                        )
                        print(
                            f"Task {idx}, Trial {i+1}/{config.num_trials}: Temperature sampling (no perturbation shift), "
                            f"T~U[{config.temperature_sampling_min}, {config.temperature_sampling_max}]"
                        )
            # Sample new perturbations for this trial (Best of N); skip for first attempt (i == 0)
            elif i > 0 and config.model_provider == "local_hf" and hasattr(agent, "sample_perturbations"):
                chosen_sigma = agent.sample_perturbations()
                print(f"Task {idx}, Trial {i+1}/{config.num_trials}: Sampled new perturbations")
            elif i == 0 and config.model_provider == "local_hf":
                print(f"Task {idx}, Trial {i+1}/{config.num_trials}: No bias sampling (first attempt)")

            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}, trial {i+1}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                    perturbation_sigma=chosen_sigma,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                    perturbation_sigma=chosen_sigma,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            results.append(result)

    display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    # Check if using local HuggingFace model
    if config.model_provider == "local_hf":
        if config.agent_strategy == "tool-calling":
            from tau_bench.agents.local_hf_agent import LocalHFToolCallingAgent
            return LocalHFToolCallingAgent(
                tools_info=tools_info,
                wiki=wiki,
                model_path=config.model,
                temperature=config.temperature,
                perturbation_sigma=config.perturbation_sigma,
            )
        else:
            raise ValueError(f"Agent strategy '{config.agent_strategy}' not yet supported for local_hf provider. Only 'tool-calling' is supported.")
    
    # Original API-based agents
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
