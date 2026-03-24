with open("agent_2d.py", "r") as f:
    code = f.read()

old = """        try:
            smile_signal = self.smile_detector(chain_2d)
            skew_signal = self.skew_detector(chain_2d)
            
            score = score + (smile_signal - 0.5) * 0.1
            score = score + (skew_signal - 0.5) * 0.1
            score = torch.clamp(score, 0.01, 0.99)
        except:
            pass"""

new = """        try:
            smile_signal = self.smile_detector(chain_2d)
            skew_signal = self.skew_detector(chain_2d)
            
            score = score + (smile_signal - 0.5) * 0.1
            score = score + (skew_signal - 0.5) * 0.1
            score = torch.clamp(score, 0.01, 0.99)
        except Exception as e:
            import warnings
            warnings.warn(f"smile/skew detector failed: {e}", RuntimeWarning)"""

assert old in code, "Pattern not found"
code = code.replace(old, new, 1)

with open("agent_2d.py", "w") as f:
    f.write(code)
print("Done.")
