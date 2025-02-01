import flax.struct
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp
from brax.v1.envs.env import Env, State, Wrapper


class CompletedEvalMetrics(flax.struct.PyTreeNode):
	current_episode_metrics: dict[str, jp.ndarray]
	completed_episodes_metrics: dict[str, jp.ndarray]
	completed_episodes: jp.ndarray
	completed_episodes_steps: jp.ndarray


class CompletedEvalWrapper(Wrapper):
	"""Brax env with eval metrics for completed episodes."""

	STATE_INFO_KEY = "completed_eval_metrics"

	def reset(self, rng: jp.ndarray) -> State:
		reset_state = self.env.reset(rng)
		reset_state.metrics["reward"] = reset_state.reward
		eval_metrics = CompletedEvalMetrics(
			current_episode_metrics=jax.tree.map(jp.zeros_like, reset_state.metrics),
			completed_episodes_metrics=jax.tree.map(
				lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics
			),
			completed_episodes=jp.zeros(()),
			completed_episodes_steps=jp.zeros(()),
		)
		reset_state.info[self.STATE_INFO_KEY] = eval_metrics
		return reset_state

	def step(self, state: State, action: jp.ndarray) -> State:
		state_metrics = state.info[self.STATE_INFO_KEY]
		if not isinstance(state_metrics, CompletedEvalMetrics):
			raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
		del state.info[self.STATE_INFO_KEY]
		nstate = self.env.step(state, action)
		nstate.metrics["reward"] = nstate.reward
		# steps stores the highest step reached when done = True, and then
		# the next steps becomes action_repeat
		completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
			nstate.info["steps"] * nstate.done
		)
		current_episode_metrics = jax.tree.map(
			lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
		)
		completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
		completed_episodes_metrics = jax.tree.map(
			lambda a, b: a + jp.sum(b * nstate.done),
			state_metrics.completed_episodes_metrics,
			current_episode_metrics,
		)
		current_episode_metrics = jax.tree.map(
			lambda a, b: a * (1 - nstate.done) + b * nstate.done,
			current_episode_metrics,
			nstate.metrics,
		)

		eval_metrics = CompletedEvalMetrics(
			current_episode_metrics=current_episode_metrics,
			completed_episodes_metrics=completed_episodes_metrics,
			completed_episodes=completed_episodes,
			completed_episodes_steps=completed_episodes_steps,
		)
		nstate.info[self.STATE_INFO_KEY] = eval_metrics
		return nstate


class TimeAwarenessWrapper(Wrapper):
	"""Wraps gym environments to add time in obs."""

	def __init__(self, env: Env) -> None:
		super().__init__(env)

	@property
	def observation_size(self) -> int:
		return super().observation_size + 1

	def reset(self, rng: jp.ndarray) -> State:
		state = self.env.reset(rng)
		return state.replace(obs=jp.concatenate([state.obs, jp.ones((1,))]))

	def step(self, state: State, action: jp.ndarray) -> State:
		state = self.env.step(state.replace(obs=state.obs[:-1]), action)
		return state.replace(
			obs=jp.concatenate(
				[
					state.obs,
					(jp.array([self.episode_length]) - state.info["steps"])
					/ self.episode_length,
				]
			)
		)


class ClipRewardWrapper(Wrapper):
	"""Wraps gym environments to clip the reward to be greater than 0.

	Utilisation is simple: create an environment with Brax, pass
	it to the wrapper with the name of the environment, and it will
	work like before and will simply clip the reward to be greater than 0.
	"""

	def __init__(self, env: Env) -> None:
		super().__init__(env)

	def reset(self, rng: jnp.ndarray) -> State:
		state = self.env.reset(rng)
		return state.replace(reward=jnp.clip(state.reward, a_min=0.0))

	def step(self, state: State, action: jnp.ndarray) -> State:
		state = self.env.step(state, action)
		return state.replace(reward=jnp.clip(state.reward, a_min=0.0))
