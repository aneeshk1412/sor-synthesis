import json

# Read data from file1.json
with open('pips/highway.json', 'r') as file:
    data1 = json.load(file)

# Read data from file2.json
with open('demos/repaired_samples.json', 'r') as file:
    data2 = json.load(file)

# Append data2 to data1
data1.extend(data2)

# Write the appended list back to file1.json
with open('pips/highway.json', 'w') as file:
    json.dump(data1, file)

