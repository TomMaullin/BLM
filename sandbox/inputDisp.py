import yaml
import os


with open(os.path.join(os.getcwd(),'blm_defaults.yml'), 'r') as stream:
    inputs = yaml.load(stream)

print(inputs)
