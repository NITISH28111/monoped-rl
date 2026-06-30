#!/usr/bin/env python3
import torch
from d4pg_agent import D4PGAgent

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
# Test agent creation
agent = D4PGAgent(obs_dim=13, act_dim=3, device="cuda")
print(f"✅   Actor device: {next(agent.actor.parameters()).device}")
print(f"✅   Critic1 device: {next(agent.critic1.parameters()).device}")

# Test forward pass
obs = torch.randn(4, 13).cuda()
action = agent.actor(obs)
print(f"✅   Action tensor device: {action.device}")

print("\n🎉 D4PG is GPU-ready!")