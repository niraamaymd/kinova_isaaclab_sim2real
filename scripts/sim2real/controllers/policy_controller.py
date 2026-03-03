import torch
from utils.config_loader import parse_env_config, get_physics_properties, get_robot_joint_properties

class PolicyController:
    """Controller that loads a PyTorch state_dict policy instead of TorchScript."""

    def __init__(self):
        pass

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """Load an RL-Games state_dict and robot metadata."""
        print("\n=== Policy Loading ===")
        print(f"{'Model path:':<18} {policy_file_path}")
        print(f"{'Environment path:':<18} {policy_env_path}")

        # Load agent config (env metadata)
        self.policy_env_params = parse_env_config(policy_env_path)

        # Load RL-Games checkpoint (state_dict)
        checkpoint = torch.load(policy_file_path, map_location="cpu", weights_only=False)

        # RL-Games checkpoints usually store model weights under 'model_state' or 'agent'
        # Adjust this if your .pth structure is different
        if "model_state" in checkpoint:
            self.model_state_dict = checkpoint["model_state"]
        else:
            self.model_state_dict = checkpoint  # fallback

        # Physics & robot info
        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)
        self._max_effort, self._max_vel, self._stiffness, self._damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.dof_names
        )
        self.num_joints = len(self.dof_names)

        # Optional: keep track of action scaling
        self._action_scale = checkpoint.get("action_scale", 0.5)

        print("\n=== Policy Loaded ===\n")

    def _compute_action(self, obs):
        """Compute action using the loaded RL-Games state_dict."""
        # You need to instantiate your agent/network architecture here
        # For example, if you used a simple MLP policy:
        if not hasattr(self, "policy_network"):
            from isaaclab_rl.rl_games.networks import create_mlp_policy
            obs_dim = len(obs)
            act_dim = self.num_joints
            self.policy_network = create_mlp_policy(obs_dim, act_dim)
            self.policy_network.load_state_dict(self.model_state_dict)
            self.policy_network.eval()

        import torch
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy_network(obs_tensor).view(-1).numpy()
        return action
