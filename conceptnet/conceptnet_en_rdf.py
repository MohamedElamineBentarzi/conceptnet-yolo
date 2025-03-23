import os
import configparser
import gzip
import shutil
import subprocess
import pandas as pd
import re
from tqdm import tqdm
import json 
import rdflib
from rdflib import URIRef, Literal
from rdflib.namespace import RDF
config = configparser.ConfigParser()

# Read the ini file
config.read("config.ini")

conceptnet_assertions_link = config["links"]["conceptnet_assertions"]

output_file = os.path.join(config["paths"]["data"], "conceptnet_assertions.csv.gz") 
# Download the file



subprocess.run(['curl', '-L', conceptnet_assertions_link, '-o', output_file])


print("Download complete!")

print("Uncompressing the file...")
# Uncompress the file
with gzip.open(output_file, 'rb') as f_in:
    with open(output_file.replace(".gz", ""), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Uncompression complete!")



# clean filter non english rows
def to_snake_case(text):
    # Replace non-alphanumeric characters with underscores and make it lowercase
    text = re.sub(r'[^a-zA-Z0-9]+', '_', text).lower()
    return text

def camel_to_snake(camel_case_str):
    # Insert an underscore before each uppercase letter and convert to lowercase
    snake_case_str = re.sub(r'([a-z])([A-Z])', r'\1_\2', camel_case_str).lower()
    return snake_case_str

filtered_output = os.path.join(config["paths"]["data"], "filtered_conceptnet_assertions.csv")

print("Filtering and cleaning the data...")
with open(output_file.replace(".gz", ""), 'r', encoding='utf-8') as infile, \
     open(filtered_output, 'w', encoding='utf-8') as outfile:
    
    outfile.write(f"subject;relation;object;weight\n")
    
    # Get the total number of lines in the file for tqdm progress bar
    total_lines = sum(1 for line in open(output_file.replace(".gz", ""), 'r', encoding='utf-8'))
    
    # Process each line using tqdm for progress tracking
    for line in tqdm(infile, total=total_lines, desc="Processing lines"):

        # Split the line by whitespace (assuming space or tabs between columns)
        columns = line.strip().split('\t')
        
        # Ensure there are enough columns to extract the necessary data
        if len(columns) < 5:
            continue
        
        # Extract columns
        start_node = columns[2]
        relation = columns[1]
        end_node = columns[3]
        additional_info = columns[4]

        
        # Step 1: Check if both start_node and end_node are English (contain "/c/en/")
        if '/c/en/' in start_node and '/c/en/' in end_node:
            try:
                # Step 2: Extract the weight from the JSON in additional_info column
                info = json.loads(additional_info)
                weight = info.get('weight', None)  # Default to None if weight is missing
                if weight is None:
                    print(f"Weight missing !!!")
            except json.JSONDecodeError:
                weight = None

            # Step 3: If weight is found, write the relevant data to the new file
            if weight is not None:

                start_node = to_snake_case(start_node.split('/')[3])
                end_node = to_snake_case(end_node.split('/')[3])
                relation = camel_to_snake(relation.split('/')[-1])
                
                outfile.write(f"{start_node};{relation};{end_node};{weight}\n")

data = pd.read_csv(filtered_output, sep=';')
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

# clean every line that consideres apple as a brand
# Define the rows to drop (as a list of tuples)
rows_to_drop = [
    ('apple', 'is_a', 'computer_brand'),
    ('apple', 'related_to', 'also_computer'),
    ('apple', 'related_to', 'computer'),
    ('apple', 'related_to', 'computer_ads'),
    ('apple', 'related_to', 'computer_brand'),
    ('apple', 'related_to', 'computer_company'),
    ('apple', 'related_to', 'computer_fruit'),
    ('apple', 'related_to', 'computers'),
    ('apple', 'related_to', 'fruit_computer'),
    ('apple', 'related_to', 'fruit_computers'),
    ('apple', 'related_to', 'mac_computer'),
    ('apple', 'related_to', 'macintosh_computer'),
    ('apple', 'related_to', 'mac'),
    ('apple', 'related_to', 'mac_company'),
    ('apple', 'related_to', 'mac_os'),
    ('apple', 'related_to', 'mac_pc'),
    ('apple', 'related_to', 'macintosh'),
    ('apple', 'related_to', 'macintosh_brand'),
    ('apple', 'related_to', 'macintosh_logo'),
    ('apple', 'related_to', 'i_pod'),
    ('apple', 'related_to', 'jobs'),
    ('apple', 'related_to', 'logo'),
    ('apple', 'related_to', 'company'),
    ('apple', 'related_to', 'brand'),
    ('apple', 'used_for', 'computing'),
    ('apple', 'related_to', 'application'),
    ('apple', 'related_to', 'os'),
    ('apple', 'related_to', 'pc'),
    ('apple', 'related_to', 'phone'),
    ('apple', 'related_to', 'design'),
    ('apple', 'related_to', 'faang'),
    ('apple', 'related_to', 'i'),
    ('apple', 'related_to', 'macos'),
    ('apple', 'related_to', 'command_key')
]

# Convert rows_to_drop to a DataFrame for easier comparison
drop_df = pd.DataFrame(rows_to_drop, columns=['subject', 'relation', 'object'])

# Create a boolean mask for rows that are not in rows_to_drop
mask = ~data.apply(lambda row: (row['subject'], row['relation'], row['object']) in rows_to_drop, axis=1)


# Apply the mask to filter out the specified rows
data = data[mask]
data.reset_index(drop=True, inplace=True)

print("Filtering and cleaning complete!")

print("creating rdf graph ... ")
g = rdflib.Graph()

# Define the namespace for the RDF graph
concepts = rdflib.Namespace("conceptnet/concept/")  
relations = rdflib.Namespace("conceptnet/relation/")  
metadata = rdflib.Namespace("conceptnet/metadata/")

g.bind("concepts", concepts)
g.bind("relations", relations)
g.bind("metadata", metadata)

# Define the weight predicate (e.g., hasWeight)
hasWeight = metadata['hasWeight']



# Iterate through each row in the dataframe
for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Creating RDF triples"):
    subject = concepts[row['subject']]
    relation = relations[row['relation']]
    object_ = concepts[row['object']]
    weight = Literal(row['weight'], datatype=rdflib.XSD.float)  # Convert weight to a float literal

    # Add the main triple (subject, relation, object)
    g.add((subject, relation, object_))

    # Create a unique identifier for each triple to store the weight
    triple_id = URIRef(metadata[f"triple/{subject}/{relation}/{object_}"])

    # Add the weight as a property of the subject-relation-object triple
    g.add((triple_id, RDF.type, RDF.Statement))  # Mark it as a statement
    g.add((triple_id, RDF.subject, subject))     # Add the subject
    g.add((triple_id, RDF.predicate, relation))  # Add the predicate
    g.add((triple_id, RDF.object, object_))      # Add the object
    g.add((triple_id, hasWeight, weight))        # Add the weight

# Serialize the graph in RDF format (you can choose different formats, e.g., Turtle, RDF/XML, etc.)
output_file_path = os.path.join(config["paths"]["data"], "conceptnet_en.rdf")  # Output file path
g.serialize(destination=output_file_path, format='xml')
print("RDF graph created!")
print("Cleaning up...")
# clean the files
os.remove(output_file)
os.remove(output_file.replace(".gz", ""))
os.remove(filtered_output)
print(f"RDF graph has been written to {output_file_path}")



