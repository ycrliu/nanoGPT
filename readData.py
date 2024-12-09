

with open('/Users/rl/Desktop/SUBMIT/DLS/without', 'r') as file:
    # Read each line one by one
    for line in file:
        line = line.strip()
        if not line:
            continue
        if "MATCH: " in line:
            continue
        if "N/A" in line: continue
        print(line[line.rfind(',') + 1:].strip() + ",",end=" ")

