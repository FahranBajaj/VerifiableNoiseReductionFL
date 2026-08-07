import subprocess
import toml

config_dict = toml.load("pyproject.toml")

#Regular FedAvg
config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = False
config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = False
config_dict["tool"]["flwr"]["app"]["config"]["use-feddmc"] = False
for dataset in ["WEATHER", "MNIST", "EMNIST"]:
    num_clients = 45 if dataset == "WEATHER" else 100
    config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
    with open("pyproject.toml", 'w') as f:
        toml.dump(config_dict, f)

    subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)

#FedDMC
config_dict["tool"]["flwr"]["app"]["config"]["use-feddmc"] = True
for dataset in ["WEATHER", "MNIST", "EMNIST"]:
    num_clients = 45 if dataset == "WEATHER" else 100
    config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
    with open("pyproject.toml", 'w') as f:
        toml.dump(config_dict, f)

    subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)

#FedDMC + Standard DP
config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
for dataset in ["WEATHER", "MNIST", "EMNIST"]:
    num_clients = 45 if dataset == "WEATHER" else 100
    config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
    with open("pyproject.toml", 'w') as f:
        toml.dump(config_dict, f)

    subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)
