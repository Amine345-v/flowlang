with open('flowlang/runtime.py', 'r', encoding='utf8') as f:
    lines = f.readlines()

# Local function to fix a range of lines
def fix_range(st_idx, end_idx, base_indent):
    for i in range(st_idx, end_idx + 1):
        if i >= len(lines): break
        stripped = lines[i].lstrip()
        if not stripped:
            lines[i] = "\n"
            continue
        # We assume 4-space increments for the relative logic
        curr_stripped = lines[i].lstrip()
        # This is very hard without knowing the original intent.
        # Let's just manually re-write the broken section.
        pass

# Manually rewriting the complex _exec_action part that is broken
new_section = """
                if item.chain_node:
                    for cname, cinfo in self.chains.items():
                        if item.chain_node in cinfo["nodes"]:
                            eff = cinfo["effects"].get(item.chain_node)
                            if eff in ("satisfied", "skip", "fixed"):
                                skip_reason = str(eff)
                                break
                
                if skip_reason:
                    self.log(f"[{team}.{verb}] Exclusive Activity: Skipping '{item.chain_node}' (state={skip_reason})")
                    res_val = item.payload
                    member_idx = -1
                else:
                    if item.process_node:
                        for pname, pinfo in self.processes.items():
                            if item.process_node in pinfo["nodes"]:
                                kwargs["maestro_path"] = self._get_binary_path(pname, item.process_node)
                                break
                    
                    item.log_activity(team, verb, "Processing...")
                    
                    if self.dry_run:
                        self.log(f"[dry_run] Skip {team}.{verb}")
                        res_val = TypedValue(ValueTag.Unknown, meta={"text": "dry_run"})
                        member_idx = self._select_team_member(team)
                    else:
                        res_val, member_idx = self._execute_single_action(team, verb, i_args, kwargs, ctx)
                    
                    item.log_activity(team, verb, res_val, member_idx=member_idx)
                    
                    if item.chain_node:
                        for cname, cinfo in self.chains.items():
                            if item.chain_node in cinfo["nodes"]:
                                self._chain_call(cname, "propagate", [item.chain_node, "satisfied"], {}, ctx)
                                break
                    
                    if item.process_node:
                        for pname, pinfo in self.processes.items():
                            if item.process_node in pinfo["nodes"]:
                                pinfo["marks"][item.process_node] = f"Accomplished: {team}.{verb}"
                                self.log(f"[maestro] Mapped Order to Process Branch: {pname}/{item.process_node}")
                                break
"""

# Replace lines around 1086-1132 with new_section
# We need to find the actual line indices.
start_marker = "if item.chain_node:"
end_marker = "results.append(res_val)"

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if start_marker in line and i > 1000 and start_idx == -1:
        start_idx = i
    if end_marker in line and i > 1100 and end_idx == -1:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    lines[start_idx:end_idx] = [new_section]
    with open('flowlang/runtime.py', 'w', encoding='utf8') as f:
        f.writelines(lines)
    print(f"Fixed range {start_idx} to {end_idx}")
else:
    print(f"Failed to find markers: {start_idx}, {end_idx}")
