with open("agent_2d.py", "r") as f:
    code = f.read()

# Fix Agent2D defaults
code = code.replace(
    "    n_greeks: int = 5,\n        n_strikes: int = 20,\n        n_timesteps: int = 20,\n        base_channels: int = 32\n    ):\n        super().__init__()\n        \n        self.n_greeks = n_greeks\n        self.n_strikes = n_strikes\n        self.n_timesteps = n_timesteps",
    "    n_greeks: int = 5,\n        n_strikes: int = 30,\n        n_timesteps: int = 30,\n        base_channels: int = 32\n    ):\n        super().__init__()\n        \n        self.n_greeks = n_greeks\n        self.n_strikes = n_strikes\n        self.n_timesteps = n_timesteps"
)

with open("agent_2d.py", "w") as f:
    f.write(code)
print("Done.")
