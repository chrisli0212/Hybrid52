with open("agent_2d.py", "r") as f:
    code = f.read()

old = '        if chain_2d is None:\n            chain_2d = self._create_synthetic_chain(batch_size, device)'

new = '        if chain_2d is None:\n            import warnings\n            warnings.warn(\n                "Agent2D received chain_2d=None — no real chain data. Build chain_2d.npy first.",\n                RuntimeWarning, stacklevel=2\n            )\n            chain_2d = self._create_synthetic_chain(batch_size, device)'

count = code.count(old)
print(f"Found pattern {count} time(s)")
code = code.replace(old, new)

with open("agent_2d.py", "w") as f:
    f.write(code)
print("Done.")
