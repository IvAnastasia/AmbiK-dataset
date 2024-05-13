# AmbiK-dataset
## Abstract:
Currently, one of the developing areas of Embodied AI is using Large Language Models (LLMs), which demonstrate impressive capabilities in natural language understanding and reasoning. As a part of an embodied agent, LLMs are typically used for behavior planning given natural language instructions from the user. However, dealing with ambiguous instructions in real-world environments remains a challenge for LLMs. Various methods for task disambiguation have been proposed. However, it is difficult to compare them because they work with different data. To be able to compare different approaches and further advance this area of research, a specialized benchmark is needed. 

We propose AmbiK, the fully textual dataset of ambiguous commands addressed to a robot in a kitchen environment. AmbiK was collected with the assistance of LLMs and is human-validated. It comprises 250 pairs of ambiguous tasks and their unambiguous counterparts, categorized by ambiguity type (human preference, common sense knowledge, safety), with environment descriptions, clarifying questions and answers, and task plans, for a total of 500 tasks.

## Data collection
1. Listing the possible objects in the environment grouped by objects' similatory (e.g. different types of yogurt constitute one group).
  
2. Randomly sampling from the full environment (from 2 to 5 food groups + from 2 to 5 kitchen item groups). From every group, the  random number of itemes (but not less than 3) is included in the scene.
Getting scenes like:
- [environment_short:] garlic press, corkscrew, sunflower oil, olive oil, coconut oil, bell pepper, carrot, tomato, uncut white bread, sliced whole wheat bread
- [environment_full (in natural language):] a garlic press, a corkscrew, sunflower oil, olive oil, coconut oil, a bell pepper, a carrot, a tomato, uncut white bread, sliced whole wheat bread
  
3. the Mistral

>Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. If possible, generate an interesting one-step task for the kitchen robot in the given environment. The task should not be ambiguous. You can mention only food and objects that are in the kitchen. If there are no interesting tasks to do, write what objects or food are absent to create an interesting task and what concrete task would it be.

4. For every unambiguous task, asking ChatGPT to come up with ambiguous tasks with three different prompts:
   > Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. The task for the robot is: {the task}. Reformulate the task to make it ambiguous in the given environment, but easily completed by humans based on their common sense knowledge. Change as few words as possible. Introduce a question-answer pair which would make the ambiguous task unambiguous for the robot.'
6. Manually reviewing Mistral's and ChatGPT's answers.
