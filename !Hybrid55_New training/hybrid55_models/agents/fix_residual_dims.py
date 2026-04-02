with open("agent_2d.py", "r") as f:
    code = f.read()

old = "        n_strikes: int = 20,\n        n_timesteps: int = 20,"
new = "        n_strikes: int = 30,\n        n_timesteps: int = 30,"

assert old in code, "Pattern not found"
code = code.replace(old, new, 1)

with open("agent_2d.py", "w") as f:
    f.write(code)
print("Done.")
