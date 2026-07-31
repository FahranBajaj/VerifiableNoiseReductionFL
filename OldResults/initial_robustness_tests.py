import subprocess
import toml

config_dict = toml.load("pyproject.toml")

for attack_type in ["LIT", "GAUSSIAN", "LABELFLIP", "ADAPTIVE"]:
    config_dict["tool"]["flwr"]["app"]["config"]["attack-type"] = attack_type
    for dataset in ["WEATHER", "MNIST", "EMNIST"]:
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        num_clients = 45 if dataset == "WEATHER" else 100

        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)