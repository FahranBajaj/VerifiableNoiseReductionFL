import subprocess
import toml

config_dict = toml.load("pyproject.toml")
#No dp
config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = False
for dataset in ["MNIST", "EMNIST", "WEATHER"]:
    num_clients = 45 if dataset == "WEATHER" else 100
    config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
    with open("pyproject.toml", 'w') as f:
        toml.dump(config_dict, f)

    subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=1\"", shell = True)

for epsilon in [1,2,4,8]:
    config_dict["tool"]["flwr"]["app"]["config"]["epsilon"] = epsilon

    #ours
    config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
    config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = True
    for dataset in ["MNIST", "EMNIST", "WEATHER"]:
        num_clients = 45 if dataset == "WEATHER" else 100
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=1\"", shell = True)

    #no noise reduction
    config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
    config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = False
    for dataset in ["MNIST", "EMNIST", "WEATHER"]:
        num_clients = 45 if dataset == "WEATHER" else 100
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=1\"", shell = True)