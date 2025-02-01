import functools
import pickle
import time

import hydra
import jax
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt

from fontTools.varLib.interpolatableHelpers import min_cost_perfect_bipartite_matching
from kheperax.tasks.config import KheperaxConfig
from kheperax.tasks.main import KheperaxTask
from omegaconf import DictConfig, OmegaConf
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.populations.adaptive_centroids_population import (
	AdaptiveCentroidsPopulation,
)
from qdax.core.populations.adaptive_population import AdaptivePopulation
from qdax.core.populations.grid_population import GridPopulation, compute_cvt_centroids
from qdax.core.populations.threshold_population import ThresholdPopulation

from qdax.tasks.arm import arm_scoring_function
from qdax.tasks.brax_envs import (
	reset_based_scoring_function_brax_envs as scoring_function,
)
from qdax.tasks.environments import behavior_descriptor_extractor
from qdax.tasks.kheperax_envs import create_kheperax_energy_task
from qdax.tasks.standard_functions import (
	rastrigin_scoring_function,
	sphere_scoring_function,
)
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results
from utils import get_env
from qdax.custom_types import Metrics

from functools import partial

threshold_pop_l_value = {
	"walker2d_uni": jnp.sqrt(1/1024)/2,
	"ant_omni": jnp.sqrt((60**2)/1024)/2,
	"ant_omni_nobound": jnp.sqrt((100**2)/1024)/2, # seems like can reasonably reach -50, 50
	"antblock_omni": jnp.sqrt((60**2)/1024)/2,
	"antblockmany_omni": jnp.sqrt((60**2)/1024)/2,
}

def plot_scatterplot(repertoire: GridPopulation, min_bd: jnp.ndarray, max_bd: jnp.ndarray) -> None:
	#get every individual in the repertoire
	individuals = repertoire.genotypes

	#get the behavior descriptors of every individual
	descriptors = repertoire.descriptors

	#get the fitness of every individual
	fitnesses = repertoire.fitnesses

	#scatter plot of all the descriptors in the BD space (2d), and color code by fitness
	plt.scatter(descriptors[:, 0], descriptors[:, 1], c=fitnesses, cmap="viridis")
	plt.colorbar()
	plt.xlabel("BD 1")
	plt.ylabel("BD 2")
	plt.title(f"Scatter plot of the {type(repertoire).__name__} over the BD space")
	return plt

def plot_kheperax_maze(population: AdaptiveCentroidsPopulation, min_bd: jnp.ndarray, max_bd: jnp.ndarray, num_bd_points: int) -> None:
	#population.descriptors is a 2d array of shape (num_individuals, num_points_desc) where num_points_desc is (x_1, y_1, x_2, y_2, ...)
    #we want to plot the positions of the individuals on the maze after a number of timesteps (i.e. x_i, y_i)
    colors = plt.cm.magma(jnp.linspace(0, 1, num_bd_points // 2))
    for i in range(0, num_bd_points, 2):
        plt.scatter(population.descriptors[:, i], population.descriptors[:, i+1], 
                    color=colors[i // 2], label=f'Time {i // 2 + 1}', s=10, alpha=0.7)

    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title(f"Scatter plot of the {type(population).__name__} over the BD space")
    plt.gcf().set_dpi(300)  # Increase resolution of the image
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    return plt


def population_metrics(repertoire: GridPopulation, qd_offset: float, centroids: jnp.ndarray) -> Metrics:
    #add regular metrics
	qd_metrics = default_qd_metrics(repertoire, qd_offset)
	
	# takes repertoire, and casts it into a grid archive, and uses this for metrics
	grid_pop = GridPopulation.init(
		genotypes=repertoire.genotypes,
		fitnesses=repertoire.fitnesses,
		descriptors=repertoire.descriptors,
		centroids=centroids,
	)
	
	proj_metrics_dict = default_qd_metrics(grid_pop, qd_offset)
	proj_metrics = {f"proj_{key}": value for key, value in proj_metrics_dict.items()}

	#novelty score
	# Neighbors
	is_empty = repertoire.fitnesses == -jnp.inf
	is_neighbor = (~is_empty)[:, None] & (~is_empty)[None, :]
	is_neighbor = jnp.fill_diagonal(is_neighbor, False, inplace=False)

	# Distance to neighbors
	distance = jnp.linalg.norm(repertoire.descriptors[:, None, :] - repertoire.descriptors[None, :, :], axis=-1)
	distance = jnp.where(is_neighbor, distance, jnp.inf)
	values, indices = jax.vmap(partial(jax.lax.top_k, k=3))(-distance)
	novelty = jnp.mean(-values, where=jnp.take_along_axis(is_neighbor, indices, axis=1), axis=-1)

	metrics = {**proj_metrics, **qd_metrics, "mean_novelty": jnp.mean(novelty), "max_novelty": jnp.max(novelty), "median_novelty": jnp.median(novelty)}

	if isinstance(repertoire, AdaptiveCentroidsPopulation):
		grid_pop_just_exploit = GridPopulation.init(
			genotypes=repertoire.only_in_cells_genotypes,
			fitnesses=repertoire.only_in_cells_fitnesses,
			descriptors=repertoire.only_in_cells_descriptors,
			centroids=centroids,
		)

		proj_metrics_dict_just_exploit = default_qd_metrics(grid_pop_just_exploit, qd_offset)
		proj_metrics_just_exploit = {f"proj_exploit_{key}": value for key, value in proj_metrics_dict_just_exploit.items()}
		metrics = {**metrics, **proj_metrics_just_exploit}
	else:
		#else just copy the same metrics as the regular ones
		proj_metrics_dict_just_exploit = proj_metrics_dict
		proj_metrics_just_exploit = {f"proj_exploit_{key}": value for key, value in proj_metrics_dict_just_exploit.items()}
		metrics = {**metrics, **proj_metrics_just_exploit}


	print(f"metrics coverage {metrics['proj_coverage']}\n proj_coverage: {metrics['coverage']}")

	return metrics


@hydra.main(version_base=None, config_path="configs/", config_name="me")
def main(config: DictConfig) -> None:
	wandb.init(
		project="Adaptive-Archive",
		name="MAP-Elites",
		config=OmegaConf.to_container(config, resolve=True),
	)

	# Init a random key
	random_key = jax.random.key(config.seed)

	# Init environment
	if config.env.type == "mujoco":
		env = get_env(config)
		print(f"behavior descriptor length: {env.behavior_descriptor_length}")
		num_descriptors = env.behavior_descriptor_length
		reset_fn = jax.jit(env.reset)

		# Init policy network
		policy_layer_sizes = config.algo.policy_hidden_layer_sizes + (env.action_size,)
		policy_network = MLP(
			layer_sizes=policy_layer_sizes,
			kernel_init=jax.nn.initializers.lecun_uniform(),
			final_activation=jnp.tanh,
		)

		# Init population of controllers
		random_key, subkey = jax.random.split(random_key)
		keys = jax.random.split(subkey, num=config.population.max_size)
		fake_batch_obs = jnp.zeros(shape=(config.population.max_size, env.observation_size))
		init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

		# Print the number of parameters in the policy network
		param_count = sum(x[0].size for x in jax.tree.leaves(init_params))
		print("Number of parameters in policy_network: ", param_count)

		# Init policy network
		policy_layer_sizes = config.algo.policy_hidden_layer_sizes + (env.action_size,)
		policy_network = MLP(
			layer_sizes=policy_layer_sizes,
			kernel_init=jax.nn.initializers.lecun_uniform(),
			final_activation=jnp.tanh,
		)

		# Define the fonction to play a step with the policy in the environment
		def play_step_fn(env_state, policy_params, random_key):
			actions = policy_network.apply(policy_params, env_state.obs)
			state_desc = env_state.info["state_descriptor"]
			next_state = env.step(env_state, actions)

			transition = QDTransition(
				obs=env_state.obs,
				next_obs=next_state.obs,
				rewards=next_state.reward,
				dones=next_state.done,
				truncations=next_state.info["truncation"],
				actions=actions,
				state_desc=state_desc,
				next_state_desc=next_state.info["state_descriptor"],
			)

			return next_state, policy_params, random_key, transition

		# Prepare the scoring function
		bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
		scoring_fn = functools.partial(
			scoring_function,
			episode_length=config.env.episode_length,
			play_reset_fn=reset_fn,
			play_step_fn=play_step_fn,
			behavior_descriptor_extractor=bd_extraction_fn,
		)
		min_bd = config.env.min_bd
		max_bd = config.env.max_bd

	elif config.env.type == "kheperax":
		if config.env.name == "maze_min_energy":
			# Define Task configuration
			config_kheperax = KheperaxConfig.get_default_for_map(config.env.map_name)
			config_kheperax.episode_length = config.env.episode_length
			config_kheperax.mlp_policy_hidden_layer_sizes = tuple(config.env.mlp_policy_hidden_layer_sizes)
			config_kheperax.robot = config_kheperax.robot.replace(
				lasers_return_minus_one_if_out_of_range=config.env.lasers_return_minus_one_if_out_of_range,
			)

			# Create Kheperax Task.
			random_key, subkey = jax.random.split(random_key)
			(
				env,
				policy_network,
				scoring_fn,
			) = create_kheperax_energy_task(
				config_kheperax,
				num_points_desc=config.env.num_points_desc,
			)

			min_bd, max_bd = env.behavior_descriptor_limits
			print(f"min_bd: {min_bd}, max_bd: {max_bd}")	

			# Initialise population of controllers
			random_key, subkey = jax.random.split(random_key)
			keys = jax.random.split(subkey, num=config.algo.batch_size)
			fake_batch = jnp.zeros(shape=(config.algo.batch_size, env.observation_size))
			init_params = jax.vmap(policy_network.init)(keys, fake_batch)
		else:
			raise ValueError("Unknown Kheperax task")

		min_bd, max_bd = env.behavior_descriptor_limits
		if config.env.num_points_desc is not None:
			min_bd = jnp.concatenate([jnp.asarray(min_bd) for _ in range(config.env.num_points_desc)])
			max_bd = jnp.concatenate([jnp.asarray(max_bd) for _ in range(config.env.num_points_desc)])
			print("min_bd: ", min_bd)
			print("max_bd: ", max_bd)

		if config.env.num_points_desc is None:
			num_descriptors = 2
		else:
			num_descriptors = config.env.num_points_desc * 2

	elif config.env.type == "standard":
		if config.env.name == "rastrigin":
			scoring_fn = rastrigin_scoring_function
		elif config.env.name == "sphere":
			scoring_fn = sphere_scoring_function
		elif config.env.name == "arm":
			scoring_fn = arm_scoring_function
		else:
			raise ValueError("Unknown standard function")

		random_key, subkey = jax.random.split(random_key)
		init_params = jax.random.uniform(
			subkey,
			shape=(config.population.max_size, config.env.num_param_dimensions),
			minval=config.env.min_param,
			maxval=config.env.max_param,
		)
		min_bd = config.env.min_bd
		max_bd = config.env.max_bd
	else:
		raise ValueError("Unknown environment type")

	if config.population.name == "grid_population":  # don't need centroids for adaptive populations
		if config.env.type == "mujoco":
			# Compute the centroids
			centroids, random_key = compute_cvt_centroids(
				num_descriptors=env.behavior_descriptor_length,
				num_init_cvt_samples=config.population.n_init_cvt_samples,
				num_centroids=config.population.max_size,
				minval=min_bd,
				maxval=max_bd,
				random_key=random_key,
			)
		elif config.env.type == "standard":
			random_key, subkey = jax.random.split(random_key)
			centroids, random_key = compute_cvt_centroids(
				num_descriptors=2,  # 2 is constant for the standard functions
				num_init_cvt_samples=config.population.n_init_cvt_samples,
				num_centroids=config.population.max_size,
				minval=min_bd,
				maxval=max_bd,
				random_key=random_key,
			)
		elif config.env.type == "kheperax":
			random_key, subkey = jax.random.split(random_key)
			
			centroids, random_key = compute_cvt_centroids(
				num_descriptors=num_descriptors,  # 2 is constant for kheperax environments
				num_init_cvt_samples=config.population.n_init_cvt_samples,
				num_centroids=config.population.max_size,
				minval=min_bd,
				maxval=max_bd,
				random_key=random_key,
			)
			
		else:
			raise ValueError("Unknown environment type")

	if config.env.type == "standard" and config.population.name == "adaptive_centroids_population": #don't need centroids for adaptive archives
		random_key, subkey = jax.random.split(random_key)
		centroids, random_key = compute_cvt_centroids(
			num_descriptors=2, #2 is constant for the standard functions
			num_init_cvt_samples=config.population.n_init_cvt_samples,
			num_centroids=config.population.max_size,
			minval=config.env.min_bd,
			maxval=config.env.max_bd,
			random_key=random_key,
		)

	metrics_centroids, random_key = compute_cvt_centroids(
		num_descriptors=num_descriptors, #2 is constant for the standard functions
		num_init_cvt_samples=50_000,
		num_centroids=1024,
		minval=config.env.min_bd,
		maxval=config.env.max_bd,
		random_key=random_key,
	)

	resample_centroids = config.algo.resample_centroids 
	print(f"RESAMPLE CENTROIDS?: {resample_centroids}")
	if config.population.name == "grid_population" and not resample_centroids:
		metrics_centroids = centroids #use the same centroids for the metrics as for the archive

	@jax.jit
	def evaluate_population(random_key, population):
		population_empty = population.fitnesses == -jnp.inf

		fitnesses, descriptors, extra_scores, random_key = scoring_fn(
			population.genotypes, random_key
		)

		# Compute population QD score
		qd_score = jnp.sum((1.0 - population_empty) * fitnesses).astype(float)

		# Compute population desc error mean
		error = jnp.linalg.norm(population.descriptors - descriptors, axis=1)
		dem = (
			jnp.sum((1.0 - population_empty) * error) / jnp.sum(1.0 - population_empty)
		).astype(float)

		return random_key, qd_score, dem

	def get_elites(metric):
		return jnp.sum(metric, axis=-1)

	# Get minimum reward value to make sure qd_score are positive
	reward_offset = 0

	if config.env.type == "mujoco":
		# Define a metrics function
		metrics_function = functools.partial(
			population_metrics,
			qd_offset=reward_offset * config.env.episode_length,
			centroids=metrics_centroids,
		)
	elif config.env.type == "standard":
		metrics_function = functools.partial(population_metrics, qd_offset = config.env.qd_offset, centroids=metrics_centroids)
	elif config.env.type == "kheperax":
		metrics_function = functools.partial(population_metrics, qd_offset = 0.5, centroids=metrics_centroids)
	else:
		raise ValueError("Unknown environment type")
	

	# Define emitter
	variation_fn = functools.partial(
		isoline_variation,
		iso_sigma=config.algo.iso_sigma,
		line_sigma=config.algo.line_sigma,
		# minval=config.env.min_param,
		# maxval=config.env.max_param,
	)

	mixing_emitter = MixingEmitter(
		mutation_fn=None,
		variation_fn=variation_fn,
		variation_percentage=1.0,
		batch_size=config.algo.batch_size,
	)

	fitnesses, descriptors, extra_scores, random_key = scoring_fn(
		init_params, random_key
	)

	#print keys of extra keys
	print(extra_scores.keys())

	if config.population.name == "grid_population":
		population = GridPopulation.init(
			genotypes=init_params,
			fitnesses=fitnesses,
			descriptors=descriptors,
			centroids=centroids,
			extra_scores=extra_scores,
		)
	elif config.population.name == "adaptive_population":
		population = AdaptivePopulation.init(
			genotypes=init_params,
			fitnesses=fitnesses,
			descriptors=descriptors,
			observations=extra_scores,
			max_size=config.population.max_size,
			k=config.population.knn,
		)
	elif config.population.name == "threshold_population":
		if config.population.l_value == "auto":
			l_value = threshold_pop_l_value[config.env.name]
		else:
			l_value = config.population.l_value
		print(f"l_value: {l_value}")
		print(f"fitnesses.shape: {fitnesses.shape}")
		print(f"descriptors.shape: {descriptors.shape}")
		print(f"env.observation_size: {env.observation_size}")
		population = ThresholdPopulation.init(
			genotypes=init_params,
			fitnesses=fitnesses,
			descriptors=descriptors,
			observations=jnp.zeros(shape=(fitnesses.shape[0], env.observation_size)),
			max_size=config.population.max_size,
			l_value=l_value,
		)
	elif config.population.name == "adaptive_centroids_population":
		population = AdaptiveCentroidsPopulation.init(
			genotypes=init_params,
			fitnesses=fitnesses,
			descriptors=descriptors,
			extra_scores=extra_scores,
		)
	else:
		raise ValueError("Unknown population type")

	# Instantiate MAP-Elites
	map_elites = MAPElites(
		scoring_function=scoring_fn,
		emitter=mixing_emitter,
		metrics_function=metrics_function,
	)

	# Compute initial population and emitter state
	population, emitter_state, random_key = map_elites.init(
		population, init_params, random_key
	)

	metrics = dict.fromkeys(
		[
			"iteration",
			"qd_score",
			"coverage",
			"max_fitness",
			"mean_fitness",
			"qd_score_population",
			"dem_population",
			"time",
			"mean_novelty",
			"max_novelty",
			"median_novelty",
			"proj_qd_score",
			"proj_coverage",
			"proj_max_fitness",
			"proj_mean_fitness",
			"proj_exploit_qd_score",
			"proj_exploit_coverage",
			"proj_exploit_max_fitness",
			"proj_exploit_mean_fitness",
		],
		jnp.array([]),
	)
	csv_logger = CSVLogger("./log.csv", header=list(metrics.keys()))

	# Main loop
	map_elites_scan_update = map_elites.scan_update
	num_loops = int(config.num_generations / config.log_interval)
	for i in range(num_loops):
		start_time = time.time()
		(
			(
				population,
				emitter_state,
				random_key,
			),
			current_metrics,
		) = jax.lax.scan(
			map_elites_scan_update,
			(population, emitter_state, random_key),
			(),
			length=config.log_interval,
		)
		timelapse = time.time() - start_time

		# Metrics
		random_key, qd_score_population, dem_population = evaluate_population(
			random_key, population
		)

		current_metrics["iteration"] = jnp.arange(
			1 + config.log_interval * i,
			1 + config.log_interval * (i + 1),
			dtype=jnp.int32,
		)
		current_metrics["time"] = jnp.repeat(timelapse, config.log_interval)
		current_metrics["qd_score_population"] = jnp.repeat(
			qd_score_population, config.log_interval
		)
		current_metrics["dem_population"] = jnp.repeat(
			dem_population, config.log_interval
		)

		metrics = jax.tree.map(
			lambda metric, current_metric: jnp.concatenate(
				[metric, current_metric], axis=0
			),
			metrics,
			current_metrics,
		)

	
		best_fitness_index = jnp.argmax(population.fitnesses)
		best_fitness = population.fitnesses[best_fitness_index]
		best_descriptor = population.descriptors[best_fitness_index]

		# Log
		log_metrics = jax.tree.map(lambda metric: metric[-1], metrics)
		csv_logger.log(log_metrics)
		wandb.log(log_metrics)

	# Metrics
	with open("./metrics.pickle", "wb") as f:
		pickle.dump(metrics, f)

	# Population
	with open("./population.pickle", "wb") as f:
		pickle.dump(population, f)

	if config.env.type == "mujoco":
		# Plot
		if env.behavior_descriptor_length == 2:
			env_steps = (
				jnp.arange(config.num_generations)
				* config.env.episode_length
				* config.algo.batch_size
			)
			# get projection to grid archive
			grid_archive = GridPopulation.init(
				genotypes=population.genotypes,
				fitnesses=population.fitnesses,
				descriptors=population.descriptors,
				centroids=metrics_centroids,
			)
			fig, _ = plot_map_elites_results(
				env_steps=env_steps,
				metrics=metrics,
				repertoire=grid_archive,
				min_bd=min_bd,
				max_bd=max_bd,
			)
			fig.savefig("./plot.png")
			#log image to wandb
			wandb.log({"plot": wandb.Image(fig)})
			print("plot saved")

			#plot scatterplot of the archive over the entire BD space
			fig1, _ = plot_scatterplot(
				repertoire=population,
				min_bd=min_bd,
				max_bd=max_bd,
			)
			fig1.savefig("./scatterplot.png")
			print("scatterplot saved")
			#log image to wandb
			wandb.log({"scatterplot": wandb.Image(fig1)})

	elif config.env.type == "standard":
		pass  # TODO: implement plotting for standard functions
	elif config.env.type == "kheperax":
		#for kheperax, we plot a 2d scatterplot of the final archive on the maze.

		#we rollout the entire archive, and plot the positions after 10, 20, ... timesteps

		fig = plot_kheperax_maze(
			population=population,
			min_bd=min_bd,
			max_bd=max_bd,
			num_bd_points=config.env.num_points_desc * 2,
		)
		fig.savefig("./kheperax_maze.png")
		wandb.log({"kheperax_maze": wandb.Image(fig)})

if __name__ == "__main__":
	main()
