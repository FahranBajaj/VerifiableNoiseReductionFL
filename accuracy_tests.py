import subprocess
import toml

for epsilon in [1,2,4,8]:
    config_dict = toml.load("pyproject.toml")
    config_dict["tool"]["flwr"]["app"]["config"]["epsilon"] = epsilon

    #No dp
    config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = False
    for dataset in ["MNIST", "EMNIST", "WEATHER"]:
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run("flwr run . --stream --federation-config \"num-supernodes=100 client-resources-num-cpus=1\"", shell = True)

    #ours
    config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
    config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = True
    for dataset in ["MNIST", "EMNIST", "WEATHER"]:
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run("flwr run . --stream --federation-config \"num-supernodes=100 client-resources-num-cpus=1\"", shell = True)

    #no noise reduction
    config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
    config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = False
    for dataset in ["MNIST", "EMNIST", "WEATHER"]:
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run("flwr run . --stream --federation-config \"num-supernodes=100 client-resources-num-cpus=1\"", shell = True)