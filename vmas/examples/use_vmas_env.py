import random
import time

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from vmas import make_env
from vmas.simulator.core import Agent
from vmas.simulator.utils import save_video


def _get_deterministic_action(agent: Agent, continuous: bool, env):
    if continuous:
        action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
    else:
        action = (
            torch.tensor([1], device=env.device, dtype=torch.long)
            .unsqueeze(-1)
            .expand(env.batch_dim, 1)
        )
    return action.clone()


def use_vmas_env(
    render: bool = False,
    save_render: bool = False,
    num_envs: int = 32,
    n_steps: int = 200,
    random_action: bool = False,
    device: str = "cuda",
    scenario_name: str = "social_navigation",
    continuous_actions: bool = True,
    visualize_render: bool = False,
    dict_spaces: bool = True,
    **kwargs,
):
    """Example function to use a vmas environment

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario_name (str): Name of scenario
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        dict_spaces (bool, optional): Weather to return obs, rewards, and infos as dictionaries with agent names.
            By default, they are lists of len # of agents
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        dict_spaces=dict_spaces,
        wrapper=None,
        seed=None,
        # Environment specific variables
        **kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace"),
            # schedule=torch.profiler.schedule(
            #     skip_first=2,
            #     wait=1,
            #     warmup=1,
            #     active=2,
            #     repeat=1,  # Reduce active profiling steps
            # ),
        ) as prof:
            for _ in range(n_steps):
                step += 1
                print(f"Step {step}")

                # VMAS actions can be either a list of tensors (one per agent)
                # or a dict of tensors (one entry per agent with its name as key)
                # Both action inputs can be used independently of what type of space its chosen
                dict_actions = random.choice([True, False])

                actions = {} if dict_actions else []
                for agent in env.agents:
                    if not random_action:
                        action = _get_deterministic_action(
                            agent, continuous_actions, env
                        )
                    else:
                        action = env.get_random_action(agent)
                    if dict_actions:
                        actions.update({agent.name: action})
                    else:
                        actions.append(action)

                obs, rews, dones, info = env.step(actions)

                # reset the environment if done at the indexes
                if any(dones):
                    for i in torch.arange(num_envs, device=device)[dones]:
                        obs[i] = env.reset_at(i)

                if render:
                    frame = env.render(
                        mode="human",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=visualize_render,
                    )
                    if save_render:
                        frame_list.append(frame)
    except Exception as e:
        print(f"Profiler failed: {e}")

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render and save_render:
        save_video(scenario_name, frame_list, fps=1 / env.scenario.world.dt)

    # Save the profiler trace to a JSON file
    prof.export_chrome_trace("trace.json")


if __name__ == "__main__":

    use_vmas_env(
        scenario_name="social_navigation",
        render=False,
        device="cuda",
        num_envs=32,
        n_steps=30,
        save_render=False,
        random_action=False,
        continuous_actions=True,
        # Environment specific
        n_agents=1,
    )
