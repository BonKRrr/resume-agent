import csv
import os

# Configuration
input_file = 'job_postings.csv'  # Change this to your actual filename
output_dir = 'data/jobs/raw'
column_name = 'description'     # The column header containing the text

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        file_path = os.path.join(output_dir, f'job_{i:04d}.txt')
        with open(file_path, 'w', encoding='utf-8') as out:
            out.write(row[column_name])
        
        if i % 100 == 0:
            print(f"Processed {i} files...")

print(f"Done! Created {i+1} text files in {output_dir}")
