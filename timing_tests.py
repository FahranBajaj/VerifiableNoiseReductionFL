import subprocess
import toml

config_dict = toml.load("pyproject.toml")

#Ours
config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = True
for dataset in ["WEATHER", "MNIST", "EMNIST"]:
    num_clients = 45 if dataset == "WEATHER" else 100
    config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
    with open("pyproject.toml", 'w') as f:
        toml.dump(config_dict, f)

    subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)