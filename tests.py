import subprocess
import toml

config_dict = toml.load("pyproject.toml")

#No dp
config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = False
config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = False
config_dict["tool"]["flwr"]["app"]["config"]["attack-type"] = ""
config_dict["tool"]["flwr"]["app"]["config"]["fraction-malicious"] = 0.0
for dataset in ["WEATHER", "MNIST", "EMNIST"]:
    num_clients = 45 if dataset == "WEATHER" else 100
    config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
    with open("pyproject.toml", 'w') as f:
        toml.dump(config_dict, f)

    subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)

for epsilon in [1,2,4,8]:
    config_dict["tool"]["flwr"]["app"]["config"]["epsilon"] = epsilon
    config_dict["tool"]["flwr"]["app"]["config"]["attack-type"] = ""
    config_dict["tool"]["flwr"]["app"]["config"]["fraction-malicious"] = 0.0    

    #accuracy tests
    for dataset in ["WEATHER", "MNIST", "EMNIST"]:
        config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
        num_clients = 45 if dataset == "WEATHER" else 100

        #ours
        config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
        config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = True
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)

        #no noise reduction
        config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
        config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = False
        with open("pyproject.toml", 'w') as f:
            toml.dump(config_dict, f)

        subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)

    #robustness tests
    config_dict["tool"]["flwr"]["app"]["config"]["fraction-malicious"] = 0.28
    for attack in ["LABELFLIP", "GAUSSIAN", "LIT", "SCALING", "ADAPTIVE"]:
        config_dict["tool"]["flwr"]["app"]["config"]["attack-type"] = attack
        for dataset in ["WEATHER", "MNIST", "EMNIST"]:
                config_dict["tool"]["flwr"]["app"]["config"]["dataset"] = dataset
                num_clients = 45 if dataset == "WEATHER" else 100
        
                #ours
                config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
                config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = True
                with open("pyproject.toml", 'w') as f:
                    toml.dump(config_dict, f)
        
                subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)
        
                #no noise reduction
                config_dict["tool"]["flwr"]["app"]["config"]["use-dp"] = True
                config_dict["tool"]["flwr"]["app"]["config"]["noise-reduction"] = False
                with open("pyproject.toml", 'w') as f:
                    toml.dump(config_dict, f)
        
                subprocess.run(f"flwr run . --stream --federation-config \"num-supernodes={num_clients} client-resources-num-cpus=2\"", shell = True)
