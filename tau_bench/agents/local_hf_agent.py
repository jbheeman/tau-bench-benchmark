# Copyright Sierra

import json
import sys
import os
import torch
import math
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_output_parser import parse_json, parse_xml


# Add the modified transformers path
sys.path.insert(0, "/data/jesh/workspace/hagent_orchestration/swe-bench-experiments/transformers/src")

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


class LocalHFToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model_path: str,
        temperature: float = 0.0,
        perturbation_sigma: float = 0.01,
    ):
        self.tools_info = tools_info
        self.tool_names = [t['function']['name'] for t in self.tools_info]
        self.wiki = wiki
        self.model_path = model_path
        self.temperature = temperature
        self.perturbation_sigma = perturbation_sigma
        
        # Hardcoded range for perturbation sigma (log-uniform sampling)

        
        # Load tokenizer and model
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # {"": 2}

        self.model.eval()
    
    def sample_perturbations(self):
        """Sample new perturbations for this trial."""
        if hasattr(self.model, 'sample_perturbations'):
            PERTURBATION_SIGMA_MIN = 0.001
            PERTURBATION_SIGMA_MAX = 0.13
            perturbation_sigma = self.perturbation_sigma
            # Sample perturbation sigma from log-uniform distribution
            sampled_sigma = 0.0
            if perturbation_sigma > 0 and hasattr(self.model, 'create_perturbation_manager'):
                # Log-uniform sampling: sample uniformly in log space
                # log_min = math.log(PERTURBATION_SIGMA_MIN)
                # log_max = math.log(PERTURBATION_SIGMA_MAX)
                log_min = PERTURBATION_SIGMA_MIN
                log_max = PERTURBATION_SIGMA_MAX
                log_sampled = torch.empty(1).uniform_(log_min, log_max).item()
                sampled_sigma = log_sampled
                print(sampled_sigma)
                
                self.model.create_perturbation_manager(sigma=sampled_sigma)
                print(f"Created perturbation manager with sampled sigma={sampled_sigma:.6f} (from log-uniform range [{PERTURBATION_SIGMA_MIN}, {PERTURBATION_SIGMA_MAX}])")
            
            self.model.sample_perturbations()
            return sampled_sigma
    
    def reset_perturbations(self):
        """Reset perturbations."""
        if hasattr(self.model, 'reset_perturbations'):
            self.model.reset_perturbations()
    
    def _call_model(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call the local model with messages and return the response message."""
        # Convert messages to format expected by tokenizer
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                formatted_messages.append({"role": "system", "content": msg.get("content", "")})
            elif role == "user":
                formatted_messages.append({"role": "user", "content": msg.get("content", "")})
            elif role == "assistant":
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Format tool calls for the model
                    formatted_messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    })
                else:
                    formatted_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Tool responses are added as tool messages
                formatted_messages.append({
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id", "")
                })
        
        # Apply chat template with tools
        text = self.tokenizer.apply_chat_template(
            formatted_messages,
            tools=self.tools_info,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse the response to extract tool calls if any
        # The tokenizer should handle tool calling format
        response_messages = self.tokenizer.apply_chat_template(
            formatted_messages + [{"role": "assistant", "content": generated_text}],
            tools=self.tools_info,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        
        # Try to extract tool calls from the generated text
        # Qwen2.5-Coder uses a specific format for tool calls
        message_dict = {"role": "assistant", "content": generated_text}
        try:
            tools = parse_json(parse_xml(generated_text))
        except:
            try:
                tools = parse_json(generated_text.replace('xml', '').replace('\n', ''))
            except:
                tools = None
        if tools:
            message_dict["tool_calls"] = [
                {
                    "id": tools['name'],
                    "type": "function",
                    "function": {
                        "name": tools['name'],
                        "arguments": json.dumps(tools['arguments'])
                    }
                }
            ]
        
        # Check if there are tool calls in the response
        # Qwen2.5-Coder may format them in a specific way
        # import re
        # tool_call_pattern = r'<tool_call>.*?</tool_call>'
        # if re.search(tool_call_pattern, generated_text, re.DOTALL):
        #     # Parse tool calls
        #     tool_calls = []
        #     for match in re.finditer(tool_call_pattern, generated_text, re.DOTALL):
        #         tool_call_text = match.group().replace('\n', '')
        #         name_match = re.search(r'<tool_call name="([^"]+)"', tool_call_text)
        #         if name_match:
        #             tool_name = name_match.group(1)
        #             # Extract arguments
        #             args_match = re.search(r'<tool_call_arguments>(.*?)</tool_call_arguments>', tool_call_text, re.DOTALL)
        #             if args_match:
        #                 try:
        #                     args = json.loads(args_match.group(1))
        #                 except:
        #                     args = {}
        #             else:
        #                 args = {}
        #             tool_calls.append({
        #                 "id": f"call_{len(tool_calls)}",
        #                 "type": "function",
        #                 "function": {
        #                     "name": tool_name,
        #                     "arguments": json.dumps(args) if isinstance(args, dict) else args_match.group(1)
        #                 }
        #             })
        #     if tool_calls:
        #         message_dict["tool_calls"] = tool_calls
        
        return message_dict
    
    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        # Perturbations should be sampled before solve() is called (at trial level)
        # This ensures all tasks in a trial use the same perturbation
        
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            # Call model
            next_message = self._call_model(messages)
            
            # Parse action from message
            if "tool_calls" in next_message and next_message["tool_calls"]:
                # Extract first tool call
                tool_call = next_message["tool_calls"][0]
                action = Action(
                    name=tool_call["function"]["name"],
                    kwargs=json.loads(tool_call["function"]["arguments"]),
                )
            else:
                # Default to respond action
                action = Action(
                    name=RESPOND_ACTION_NAME,
                    kwargs={"content": next_message.get("content", "")}
                )
            
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            if action.name != RESPOND_ACTION_NAME:
                # Keep only first tool call
                if "tool_calls" in next_message:
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message.get("tool_calls", [{}])[0].get("id", ""),
                            "name": action.name,
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                # if env_response.observation == '':
                #     break
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        
        # Don't reset perturbations here - they should persist for the entire trial
        # Reset will happen at the start of the next trial
        
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

