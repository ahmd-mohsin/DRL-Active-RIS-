import yaml

# Prompt the user for the file path
file_path = r"/home/ahmed/DRL-Active-RIS-/network/moppo.txt"

lines = []
with open(file_path, 'r') as f:
    while line := f.readline():
        lines.append(line.rstrip().split(':'))

yaml_format = yaml.dump(
    {service: {
        'image': {
            'tag': name
        }
    }
     for service, name in lines})

with open('result.yaml', 'w') as f:
    f.write(yaml_format)