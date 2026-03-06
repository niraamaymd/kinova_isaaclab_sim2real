import yaml
import re

with open("../../pretrained_models/reach/env.yaml", "r") as f:
    content = f.read()

cleaned = re.sub(r'!!python/[a-z/]+:[\w\.]+', '', content)
cleaned = re.sub(r'!!python/tuple', '', cleaned)

data = yaml.safe_load(cleaned)

with open("../../pretrained_models/reach/env.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print("Successfully cleaned!")
