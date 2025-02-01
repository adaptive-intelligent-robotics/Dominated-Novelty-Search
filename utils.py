import pickle

import jax
import jax.numpy as jnp
import pandas as pd
from jax.flatten_util import ravel_pytree
from omegaconf import OmegaConf
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.populations.grid_population import GridPopulation
from qdax.tasks import environments, environments_v1


def get_env(config):
	if config.env.version == "v1":
		if (
			config.env.name == "hopper_uni"
			or config.env.name == "walker2d_uni"
			or config.env.name == "halfcheetah_uni"
		):
			env = environments_v1.create(
				config.env.name, episode_length=config.env.episode_length
			)
		elif (
			config.env.name == "ant_uni"
			or config.env.name == "ant_omni"
			or config.env.name == "ant_omni_nobound"
			or config.env.name == "anttrap_omni"
			or config.env.name == "antblock_omni"
			or config.env.name == "antblockmany_omni"
		):
			env = environments_v1.create(
				config.env.name,
				episode_length=config.env.episode_length,
				use_contact_forces=config.env.use_contact_forces,
				exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
			)
		elif config.env.name == "humanoid_uni" or config.env.name == "humanoid_omni":
			env = environments_v1.create(
				config.env.name,
				episode_length=config.env.episode_length,
				exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
				backend=config.env.backend,
			)
		else:
			raise ValueError("Invalid environment name.")
	elif config.env.version == "v2":
		if (
			config.env.name == "hopper_uni"
			or config.env.name == "walker2d_uni"
			or config.env.name == "halfcheetah_uni"
		):
			env = environments.create(
				config.env.name,
				episode_length=config.env.episode_length,
				backend=config.env.backend,
			)
		elif (
			config.env.name == "ant_uni"
			or config.env.name == "ant_omni"
			or config.env.name == "anttrap_omni"
		):
			env = environments.create(
				config.env.name,
				episode_length=config.env.episode_length,
				use_contact_forces=config.env.use_contact_forces,
				exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
				backend=config.env.backend,
			)
		elif config.env.name == "humanoid_uni" or config.env.name == "humanoid_omni":
			env = environments.create(
				config.env.name,
				episode_length=config.env.episode_length,
				exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
				backend=config.env.backend,
			)
		else:
			raise ValueError("Invalid environment name.")
	else:
		raise ValueError("Invalid Brax version.")

	return env

def get_config(run_dir):
	config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
	return config


def get_metrics(run_dir):
	with open(run_dir / "metrics.pickle", "rb") as metrics_file:
		metrics = pickle.load(metrics_file)
	return pd.DataFrame.from_dict(metrics)


def get_log(run_dir):
	return pd.read_csv(run_dir / "log.csv")


def get_repertoire(run_dir):
	# Get config
	config = get_config(run_dir)

	# Init a random key
	random_key = jax.random.key(config.seed)

	# Init environment
	env = get_env(config)

	# Init policy network
	policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
	policy_network = MLP(
		layer_sizes=policy_layer_sizes,
		kernel_init=jax.nn.initializers.lecun_uniform(),
		final_activation=jnp.tanh,
	)

	# Init fake params
	random_key, random_subkey = jax.random.split(random_key)
	fake_obs = jnp.zeros(shape=(env.observation_size,))
	fake_params = policy_network.init(random_subkey, fake_obs)

	# Load repertoire
	_, reconstruction_fn = ravel_pytree(fake_params)

	# Return repertoire
	return GridPopulation.load(
		reconstruction_fn=reconstruction_fn, path=str(run_dir) + "/repertoire/"
	)


def get_df(results_dir):
	metrics_list = []
	for env_dir in results_dir.iterdir():
		for algo_dir in env_dir.iterdir():
			if algo_dir.name == "pga_me_old":
				continue
			for run_dir in algo_dir.iterdir():
				# Get config and metrics
				config = get_config(run_dir)
				metrics = get_metrics(run_dir)

				# Run
				metrics["run"] = run_dir.name

				# Env
				metrics["env"] = config.env.name

				# Algo
				metrics["algo"] = config.algo.name

				# Number of Evaluations
				if config.algo.name == "me_es":
					metrics["num_evaluations"] = metrics["iteration"] * 1050
				elif config.algo.name == "dcg_me":
					metrics["num_evaluations"] = metrics["iteration"] * (
						config.batch_size + config.algo.actor_batch_size
					)
				else:
					metrics["num_evaluations"] = (
						metrics["iteration"] * config.batch_size
					)

				metrics_list.append(metrics)
	return pd.concat(metrics_list, ignore_index=True)
