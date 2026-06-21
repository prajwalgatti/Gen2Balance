import json


def extract_json(text):
    """Parse a JSON object/array from a model response, tolerating ```json fences.

    Robust replacement for ``text.lstrip("```json").rstrip("```")`` — that idiom
    strips *character sets*, not the fence, and can silently eat leading j/s/o/n
    or trailing backticks from real content.
    """
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence line (``` or ```json)
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        # drop the closing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return json.loads(t.strip())


def generate_action_definition_prompt(action_class):
    """
    Generates the text part of the prompt for deconstructing an action.
    The actual videos will be passed alongside this text.
    """
    prompt = f"""# Prompt for Deconstructing a Visual Action

## ROLE
You are a highly structured expert in visual analysis and semantics. Your purpose is to deconstruct a given human action into its core components for an AI video generation pipeline.

## GOAL
For the given action class: `{action_class}`, generate a single, structured JSON object that provides a complete visual and semantic definition of the action. In cases where the action class name may have multiple interpretations, use the provided reference videos to determine the specific meaning intended.

## OUTPUT REQUIREMENTS
1.  **Format:** The output MUST be a single, raw JSON object.
2.  **Strictness:** Follow the provided JSON format. Do NOT add any explanatory text, comments, or apologies before or after the JSON object. Your entire response must be the JSON content itself.
3.  **JSON Schema:** The JSON object must conform to the following schema:
    - `"action_definition": string` - A concise, one-sentence definition that accurately describes the action using visual cues.
    - `"key_visual_elements": Array<string>` - A list of essential objects, body parts, movements, or other visual aspects observed during the action. Be specific.
    - `"common_mistakes_to_avoid": Array<string>` - A list of visually similar but incorrect actions. This requires you to infer what could be confused with the specific action shown.

## REFERENCE EXAMPLES
The following examples demonstrate the expected JSON structure and level of detail. These examples are not provided with video files. Use these for guidance on format and style:

### Example 1: brushing teeth
```json
{{
  "action_definition": "The act of moving a small handheld toothbrush with bristles back and forth across the teeth inside the mouth, to clean the teeth's surfaces.",
  "key_visual_elements": [
    "A toothbrush held in the dominant hand near the mouth.",
    "Toothpaste visible on the toothbrush bristles or around the mouth.",
    "Back-and-forth or circular motions of the toothbrush against teeth.",
    "The person's mouth slightly open to accommodate the toothbrush.",
    "A bathroom sink or mirror typically visible in the background.",
    "Foaming or bubbles around the mouth from toothpaste.",
    "The person spitting into a sink intermittently.",
    "The elbow bent with the forearm raised to bring the toothbrush to mouth height."
  ],
  "common_mistakes_to_avoid": [
    "Using a hairbrush or other type of brush on hair.",
    "Eating or drinking something.",
    "Flossing (uses string/floss rather than a brush).",
    "Using mouthwash (involves swishing liquid, not brushing motions).",
    "Brushing or grooming anything other than teeth."
  ]
}}
```

### Example 2: playing violin
```json
{{
  "action_definition": "The act of producing music where a person holds a four-stringed wooden stringed instrument against their shoulder and chin while drawing a bow across the strings with one hand and pressing the strings with fingers of the other hand.",
  "key_visual_elements": [
    "A wooden violin held between the chin and left shoulder",
    "The right arm moving a long wooden bow horizontally across the violin strings in smooth, controlled strokes",
    "The left hand positioned on the violin neck with fingers pressing down on strings at various positions",
    "The head tilted to the left side to secure the violin against the shoulder",
    "The characteristic curved body shape and f-holes of the violin",
    "Precise arm and wrist movements coordinated between both hands"
  ],
  "common_mistakes_to_avoid": [
    "Playing a cello, or double bass (different sizes and playing positions)",
    "Playing a guitar or other plucked string instrument",
    "Holding the violin like a guitar against the body",
    "Using fingers to pluck strings instead of bowing",
    "Holding the violin without playing the instrument"
  ]
}}
```

### Example 3: playing basketball
```json
{{
  "action_definition": "The act of participating in a team sport where players dribble, pass, and shoot an orange basketball toward an elevated basketball basketball hoop while moving around a court, often competing against other players.",
  "key_visual_elements": [
    "An orange, textured basketball.",
    "A basketball hoop with a backboard and net.",
    "A person bouncing (dribbling) an orange basketball repeatedly against the floor while running or walking.",
    "The act of shooting: propelling the ball towards the hoop with a specific form.",
    "Passing the ball between players."
  ],
  "common_mistakes_to_avoid": [
    "Playing with a different type of ball (e.g., a soccer ball, football or volleyball).",
    "Simply standing on a basketball court without interacting with a ball.",
    "Athletic drills on a court that do not involve a basketball.",
    "Throwing other objects into a hoop.",
    "Players jumping or running on a basketball court without a ball."
  ]
}}
```

## YOUR TASK
### Action Class: `{action_class}`
### Reference Content: Video files are provided with this prompt. Analyze the provided video files to understand the specific visual characteristics of the action class: '{action_class}'. If the action name could have multiple interpretations (e.g., 'batting' could mean both baseball or cricket), use the videos to determine the intended meaning.
### Generated JSON:
```json
"""
    return prompt



def create_action_prompt(ACTION_CLASS, action_deconstruction):
    """
    Generates a prompt for creating diverse text-to-video prompts.
    
    Args:
        ACTION_CLASS: The name of the action class
        action_deconstruction: Optional dict containing:
            - action_definition: Clear definition of the action
            - key_visual_elements: List of essential visual elements
            - common_mistakes_to_avoid: List of similar but incorrect actions
    """
    
    # Build context section based on whether deconstruction is provided
    context_section = f"""## Action Context
Based on the provided action analysis for "{ACTION_CLASS}":

**Definition:** {action_deconstruction['action_definition']}

**Essential Visual Elements to Include:**
{chr(10).join(f"- {element}" for element in action_deconstruction['key_visual_elements'])}

**Common Mistakes to Avoid (Do NOT depict):**
{chr(10).join(f"-  {mistake}" for mistake in action_deconstruction['common_mistakes_to_avoid'])}"""
    
    prompt = f"""# Instruction for Generating Diverse Text-to-Video Prompts

## ROLE
You are an expert at creating detailed, visually-specific prompts for text-to-video generation models. You understand the importance of precise visual descriptions and action clarity.

# GOAL
Generate 25 diverse, specific text prompts that could be used for a text-to-video generation model to create videos of the action "{ACTION_CLASS}".

{context_section}

## Requirements
Each prompt should:
- Explicitly mention the action class (e.g., "person is eating" if the action is "eating")
- Include a brief description of the physical movements involved in the action
- Describe the action in a specific context, with specific details
- Incorporate the key visual elements identified above naturally into the prompts
- Ensure the action depicted is clearly {ACTION_CLASS} and not any of the similar actions to avoid

Create a mix of scenarios that naturally fit the action - focus on the typical contexts where this action actually occurs (e.g., for eating: restaurants, homes, picnics; for exercising: gyms, parks, home workouts; for professional activities: offices, workshops, studios).

Vary the following aspects across the set of prompts while maintaining realistic, natural scenarios:
- Settings/environments (realistic locations where the action naturally occurs such as homes, public spaces, outdoors, workplaces, etc.)
- Camera angles and framing (close-up, wide shot, side view, shaky handheld, stable tripod, etc.)
- Video quality (professional-looking, amateur phone camera, slightly blurry, well-lit, poorly lit, etc.)
- People performing the action (different ages, demographics, clothing, single/multiple people)
- Objects involved (different tools, items, variations relevant to the action)
- Action styles (different speeds, intensities, skill levels from amateur to professional)
- Time of day and lighting conditions (daylight, evening, indoors with artificial lighting, etc.)
- Background elements (crowded, empty, urban, rural, etc.)
- Social context (alone, with family, with friends, in public, etc.)

## Format
Return exactly 25 prompts as a numbered JSON dictionary with keys "1" through "25". Each prompt should be a single sentence that is clear, specific, and detailed enough to guide a text-to-video model, typically 15-30 words. Always begin with a subject followed by the action performed (for example, "A person is eating") or the appropriate form of the action verb, and include a description of the physical movements.

## Examples

### Action Class: drinking
```json
{{
    "1": "A teenager is drinking water, tilting his head back as he raises a plastic bottle to his lips, while skateboarding in a park.",
    "2": "An elderly man is drinking hot tea, carefully lifting a chipped mug to his mouth with both hands, in his dimly lit kitchen with a cluttered counter.",
    "3": "A mother is drinking coffee, taking quick sips between conversations, from a travel mug at a playground while keeping an eye on her children, viewed from across the playground.",
    "4": "Office workers are drinking water, passing a bottle around and tipping it back to swallow, during an informal meeting in a fluorescent-lit break room.",
    "5": "A sweaty runner is drinking from a sports bottle, squeezing liquid into his mouth without touching the rim, after a race with cheering spectators in the background, captured in close-up."
}}
```

### Action Class: kicking field goal
```json
{{
    "1": "A teenage girl is kicking a field goal, taking three steps back before running forward and sending the ball between posts with her foot, on an empty school field with makeshift posts.",
    "2": "A father is teaching his young son how to kick a field goal, demonstrating the proper foot placement and swinging motion, in their backyard with the house visible behind them, seen from a side angle.",
    "3": "An amateur player is kicking a field goal, planting his non-kicking foot firmly while swinging his kicking leg through the ball, at a community game with a small crowd watching.",
    "4": "A professional football player is kicking a field goal, executing a perfect follow-through as his foot connects with the ball sending it high above the crossbar, during a televised game viewed from behind the goalposts.",
    "5": "Friends are taking turns kicking field goals, setting up the ball on a backpack and attempting to kick it between two trees, in a park on a sunny afternoon."
}}
```

### Action Class: playing guitar
```json
{{
    "1": "A young woman is playing guitar, her fingers pressing down on frets while her other hand strums the strings, alone in her bedroom with basic lighting.",
    "2": "A street performer is playing an acoustic guitar, rhythmically tapping the body while picking individual strings, in a busy pedestrian area with people walking by, shown from across the street.",
    "3": "A grandfather is playing guitar, slowly demonstrating chord changes by positioning his fingers on the fretboard, while teaching his grandchild on an old instrument.",
    "4": "A teenager is playing electric guitar, head bobbing as he rapidly moves his picking hand up and down across the strings, in a garage band practice, filmed from a low angle.",
    "5": "Fingers are playing intricate riffs on a classical guitar, precisely moving between strings and frets with controlled movements, in a close-up view."
}}
```

{f'''## Important Guidelines Based on Action Analysis
When generating prompts for "{ACTION_CLASS}":
1. Ensure each prompt clearly depicts the specific action as defined, not similar activities
2. Include relevant visual elements naturally in the scene descriptions
3. Be specific about the movements and objects involved to avoid ambiguity
4. Double-check that your prompts won't be confused with the similar actions listed above
''' if action_deconstruction else ''}

Please generate exactly 25 unique, diverse prompts for the action "{ACTION_CLASS}" following these patterns. Make sure to return exactly 25 prompts numbered from 1 to 25.

Action Class: {ACTION_CLASS}
Generated Prompts:
```json
"""
    return prompt