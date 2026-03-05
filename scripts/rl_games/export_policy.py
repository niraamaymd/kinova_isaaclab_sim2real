import torch
import yaml
from rl_games.algos_torch import model_builder

def export_policy():
    # 1. Paths
    agent_cfg_path = "/home/robotics/niraamay/kinova_isaaclab_sim2real/pretrained_models/reach/agent.yaml"
    checkpoint_path = "/home/robotics/niraamay/kinova_isaaclab_sim2real/pretrained_models/reach/policy.pth"
    
    with open(agent_cfg_path, 'r') as f:
        agent_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # ALIGNED WITH YOUR TRAINING: 25 observations, 6 actions
    num_obs = 25 
    num_actions = 6 
    
    print(f"Building model for Obs: {num_obs}, Actions: {num_actions}")

    # 2. Reconstruct the structure for RL-Games Builder
    builder_params = {
        'network': agent_cfg['params']['network'],
        'model': agent_cfg['params']['model']
    }

    builder = model_builder.ModelBuilder()
    network_builder = builder.load(builder_params)
    
    model_config = {
        'actions_num': num_actions,
        'input_shape': (num_obs,),
        'num_seqs': 1,
        'value_size': 1,
        'normalize_input': agent_cfg['params']['config'].get('normalize_input', False),
    }
    
    model = network_builder.build(model_config)
    
    # 3. Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # strict=False allows us to ignore Critic-specific keys (like value_mean_std)
    model.load_state_dict(checkpoint['model'], strict=False)
    print("Main Actor weights loaded successfully.")
    
    # Load input normalization stats (Crucial for Sim2Real)
    if model_config['normalize_input']:
        if 'running_mean_std' in checkpoint:
            model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            print("Successfully loaded Input RunningMeanStd stats.")
        else:
            print("WARNING: normalize_input is True but no stats found in checkpoint!")

    model.eval()

    # 4. Wrap for TorchScript
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, obs):
            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs' : obs,
                'rnn_states': None
            }
            # 'mus' contains the 6 deterministic actions
            output = self.model(input_dict)
            return output['mus'] 

    wrapper = PolicyWrapper(model)
    example_input = torch.randn(1, num_obs)
    
    # 5. Trace and Save
    traced_script = torch.jit.trace(wrapper, example_input)
    traced_script.save("policy_compiled.pt")
    
    print("\nSUCCESS: Exported 'policy_compiled.pt' with 6 actions.")

if __name__ == "__main__":
    export_policy()
