import csv
import os
import re

# --- CONFIGURATION ---
INPUT_CSV = 'postings.csv'
OUTPUT_DIR = 'data/jobs/raw'

# Increase limit for massive CSV cells
csv.field_size_limit(2147483647) 

def clean(text):
    if not text or text.lower() == 'nan': return ""
    # Remove HTML tags and normalize whitespace
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Reading {INPUT_CSV} and extracting full job context...")

try:
    with open(INPUT_CSV, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            # Extract multiple fields
            title = clean(row.get('title', ''))
            desc = clean(row.get('description', ''))
            skills = clean(row.get('skills_desc', ''))
            exp = clean(row.get('formatted_experience_level', ''))
            work_type = clean(row.get('formatted_work_type', ''))
            
            # Construct the comprehensive text body
            # We add labels (Title:, Skills:) to help the model distinguish sections
            parts = [
                f"Title: {title}",
                f"Experience Level: {exp}" if exp else "",
                f"Work Type: {work_type}" if work_type else "",
                f"Skills Required: {skills}" if skills else "",
                f"Description: {desc}"
            ]
            
            # Join only the parts that aren't empty
            full_text = "\n".join([p for p in parts if p])

            if desc: # Ensure we at least have a description
                job_id = row.get('job_id', str(count))
                file_path = os.path.join(OUTPUT_DIR, f'job_{job_id}.txt')
                
                with open(file_path, 'w', encoding='utf-8') as out:
                    out.write(full_text)
                count += 1
            
            if count % 1000 == 0 and count > 0:
                print(f"Processed {count} jobs...")

    print(f"\nSuccess! {count} detailed job files created in '{OUTPUT_DIR}'")

except Exception as e:
    print(f"Error during ingestion: {e}")
