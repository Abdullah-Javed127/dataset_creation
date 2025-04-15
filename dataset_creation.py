import json
import random
import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Load GROQ API Key from env variable 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"  # Or any GROQ-supported model
print(GROQ_API_KEY)
# Adjust based on your GROQ API plan
REQUESTS_PER_MINUTE = 15
ENTRIES_PER_BATCH = REQUESTS_PER_MINUTE
TOTAL_ENTRIES = 1000
OUTPUT_FILE = "scene_dataset.jsonl"

COCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

def generate_random_scene():
    return {
        "left": random.sample(COCO_OBJECTS, random.randint(0, 3)),
        "bottom": random.sample(COCO_OBJECTS, random.randint(0, 3)),
        "up": random.sample(COCO_OBJECTS, random.randint(0, 3)),
        "right": random.sample(COCO_OBJECTS, random.randint(0, 3)),
    }

def build_prompt(scene):
    return """
You are generating scene descriptions for an AI dataset based on realistic, logically consistent inputs. Each entry represents a real-world scene where objects appear in natural combinations, appropriate for the given region of the picture.

Output Format:
{
  "left": [ ... ],
  "down": [ ... ],
  "up": [ ... ],
  "right": [ ... ],
  "description": "..."
}

Object Pool (use only these):
["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

Instructions:
1. **Region Definitions:**  
   - **Up:** The region in the scene that is slightly far from the point-of-view (not the sky, but objects in the distance).  
   - **Down:** The region immediately in front and close to the point-of-view.  
   - **Left:** The left side of the picture.  
   - **Right:** The right side of the picture.

2. **Input Constraints:**  
   - Randomly place 1–4 unique objects per region.  
   - Use the object pool only.  
   - Ensure the combination is realistic; for example, on a street you might see cars, pedestrians, birds, and bicycles—not a wild animal (like an elephant) appearing next to an airplane unless the context justifies it.  
   - Avoid illogical pairings (e.g. a giraffe next to an airplane) by maintaining real-world context.

3. **Description Generation:**  
   - Create a natural language description that explains the scene logically without inventing implausible interactions.  
   - Start with the **up** region by referring to objects that are seen in the distance.  
   - Describe the **down** region as those objects near the viewer.  
   - Seamlessly integrate details of the **left** and **right** regions.  
   - Use varied, human-like phrasing and simple, realistic sentences.
   - The description must be ≤ 250 characters and use no more than 4 sentences.
   - Output only a single JSON object; do not include anything else.

Examples:
{
  "left": ["bicycle", "person"],
  "down": ["car", "bus"],
  "up": ["traffic light", "train"],
  "right": ["bench"],
  "description": "In the distance, a traffic light and train appear along the road. Directly in front, a car and bus move steadily, while a bicycle and person are visible to the left, and a bench sits to the right."
}

{
  "left": ["bird"],
  "down": ["person", "car"],
  "up": ["stop sign"],
  "right": ["bicycle", "bus"],
  "description": "A stop sign stands in the distance. In front, a person and car are clearly visible, with a bird to the left and a bicycle beside a bus on the right."
}

Now generate one such JSON entry, ensuring the assigned objects form a logical real-world scene. Start with the 'up' region and follow all instructions strictly. Do not add anything beyond the JSON object.
"""



import re

import time

def query_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs JSON scene descriptions."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    while True:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            match = re.search(r'(\{.*\})', content, re.DOTALL)
            if match:
                json_text = match.group(1)
                try:
                    return json.loads(json_text)
                except Exception as e:
                    print("❌ Error parsing JSON:", e)
                    return None
            else:
                print("❌ No JSON object found in the response.")
                return None
        elif response.status_code == 429:
            print("⚠️ Rate limit hit. Waiting before retrying...")
            time.sleep(1.5)  # Backoff time; could parse delay from error message if needed
        else:
            print(f"❌ GROQ error {response.status_code}: {response.text}")
            return None



def main():
    entries_created = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while entries_created < TOTAL_ENTRIES:
            print(f"🚀 Generating batch... Entries created so far: {entries_created}")
            batch_start_time = time.time()

            for _ in range(min(ENTRIES_PER_BATCH, TOTAL_ENTRIES - entries_created)):
                scene = generate_random_scene()
                prompt = build_prompt(scene)
                entry = query_groq(prompt)

                if entry:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    entries_created += 1
                else:
                    print("⚠️ Skipped one entry due to error.")
                time.sleep(4)  # Optional small delay between calls

            # Rate limiting: wait for 60 seconds per batch if needed
            elapsed = time.time() - batch_start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed
                print(f"🕒 Sleeping for {int(sleep_time)} seconds to respect rate limits...")
                time.sleep(sleep_time)

    print(f"✅ Done! {entries_created} entries saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
