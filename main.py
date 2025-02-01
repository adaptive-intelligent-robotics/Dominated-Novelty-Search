import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(config: DictConfig) -> None:
	if config.algo.name == "me":
		import main_me as main
	elif config.algo.name == "dcg_me":
		import main_dcg_me as main
	elif config.algo.name == "aurora":
		import main_aurora as main
	else:
		raise NotImplementedError

	main.main(config)


if __name__ == "__main__":
	main()
