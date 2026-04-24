import json
import re

def clean_ts_content(content):
    # Very simple extraction of strings from the TS file to make it more readable
    # In a real scenario, we'd use a parser, but here we just want the text
    matches = re.findall(r'"(.*?)"', content)
    return " ".join(matches)

def prepare_data():
    output_file = "data/train.txt"
    
    with open(output_file, "w") as f:
        # Initial context
        f.write("User: Who is Murad?\n")
        f.write("Assistant: Mir Mozadded Alfeshani Murad is a curious mind passionate about technology and innovation. He is skilled in React, Next.js, Node.js, MongoDB, and Python, and he enjoys exploring AI and building impactful digital solutions.\n")
        
        # Bio info
        bio_path = "/home/murad/Developer/scratch_model/myresumenextjs/src/app/data/bio.json"
        with open(bio_path, 'r') as bio_f:
            bio_data = json.load(bio_f)
            for item in bio_data:
                f.write(f"User: Tell me about {item['title']}.\n")
                f.write(f"Assistant: {item['description']}\n")
                if 'socialLinks' in item:
                    links = ", ".join([f"{l['text']}: {l['url']}" for l in item['socialLinks']])
                    f.write(f"User: How can I connect with Murad?\n")
                    f.write(f"Assistant: You can connect with him via {links}.\n")

        # Projects info
        projects_path = "/home/murad/Developer/scratch_model/myresumenextjs/src/content/projects.ts"
        with open(projects_path, 'r') as proj_f:
            content = proj_f.read()
            # Extract project details using regex for simplicity
            project_blocks = re.findall(r'\{(.*?)\}', content, re.DOTALL)
            for block in project_blocks:
                name_match = re.search(r'name:\s*"(.*?)"', block)
                desc_match = re.search(r'description:\s*"(.*?)"', block)
                tech_match = re.search(r'tech_stack:\s*\[(.*?)\]', block, re.DOTALL)
                
                if name_match:
                    name = name_match.group(1)
                    f.write(f"User: What is {name}?\n")
                    if desc_match:
                        f.write(f"Assistant: {desc_match.group(1)}\n")
                    else:
                        f.write(f"Assistant: {name} is one of Murad's projects.\n")
                    
                    if tech_match:
                        techs = tech_match.group(1).replace('"', '').replace('\n', '').strip()
                        f.write(f"User: What technologies were used in {name}?\n")
                        f.write(f"Assistant: {name} was built using {techs}.\n")

    print(f"Data prepared and saved to {output_file}")

if __name__ == "__main__":
    prepare_data()
