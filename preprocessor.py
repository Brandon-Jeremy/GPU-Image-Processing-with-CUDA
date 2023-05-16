#Read the file
file_path = "text.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()
#Sort the lines
sorted_lines = sorted(lines)
#Write the sorted lines to a new file
output_file_path = "text.txt"
with open(output_file_path, 'w') as file:
    file.writelines(sorted_lines)

print("Lines sorted successfully!")
