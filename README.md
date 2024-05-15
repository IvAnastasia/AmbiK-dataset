# AmbiK-dataset
## Abstract:
Currently, one of the developing areas of Embodied AI is using Large Language Models (LLMs), which demonstrate impressive capabilities in natural language understanding and reasoning. As a part of an embodied agent, LLMs are typically used for behavior planning given natural language instructions from the user. However, dealing with ambiguous instructions in real-world environments remains a challenge for LLMs. Various methods for task disambiguation have been proposed. However, it is difficult to compare them because they work with different data. To be able to compare different approaches and further advance this area of research, a specialized benchmark is needed. 

We propose AmbiK, the fully textual dataset of ambiguous commands addressed to a robot in a kitchen environment. AmbiK was collected with the assistance of LLMs and is human-validated. It comprises 250 pairs of ambiguous tasks and their unambiguous counterparts, categorized by ambiguity type (human preference, common sense knowledge, safety), with environment descriptions, clarifying questions and answers, and task plans, for a total of 500 tasks.

## Data collection
1. Listing the possible objects in the environment grouped by objects' similarity (e.g. different types of yogurt constitute one group).
  
2. Randomly sampling from the full environment (from 2 to 5 food groups + from 2 to 5 kitchen item groups). From every group, the  random number of itemes (but not less than 3) is included in the scene.
Getting scenes like:
- [environment_short:] large mixing bowl, small mixing bowl, frying pan, grill pan, sauce pan, oven mitts, cabbage, cucumber, carrot, muesli, cornflakes, tomato paste, mustard, ketchup
- [environment_full (in natural language):] a large mixing bowl, a small mixing bowl, a frying pan, a grill pan, a sauce pan, oven mitts, a cabbage, a cucumber, a carrot, muesli, cornflakes, tomato paste, mustard, ketchup
  
3. For every scene, asking [Mistral](https://mistral.ai) to generate an unambiguous task:

>Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. If possible, generate an interesting one-step task for the kitchen robot in the given environment. The task should not be ambiguous. You can mention only food and objects that are in the kitchen. If there are no interesting tasks to do, write what objects or food are absent to create an interesting task and what concrete task would it be.

4. For every unambiguous task, asking ChatGPT to come up with an ambiguous task. We used with three different prompts which correspond to task types:

**Preferences:**
> Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. The task for the robot is: {the task}. Reformulate the task to make it ambiguous in the given environment. Change as few words as possible. Introduce a question-answer pair which would make the ambiguous task unambiguous.

**Common sense knowledge:**
> Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. The task for the robot is: {the task}. Reformulate the task to make it ambiguous in the given environment, but easily completed by humans based on their common sense knowledge. Change as few words as possible. Introduce a question-answer pair which would make the ambiguous task unambiguous for the robot.'

**Safety:**
> Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. The task for the robot is: {the task}. Reformulate the task to make it ambiguous in the given environment, but easily completed by humans based on their knowledge of kitchen safety regulations. Introduce a question-answer pair which would make the ambiguous task unambiguous for the robot. A question should be asked by the robot.

5. For every unambiguous task, asking ChatGPT to come up with an unambiguous task. We used the following prompt:
  > Imagine there is a kitchen robot. In the kitchen, there is also a fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle. Apart from that, in the kitchen there is {scene in natural language}. Other objects do not exist in the environment. The task for the robot is: {the task}. Please formulate the task in other words. Replace as many words as possible. You can use pronouns, hyponyms, synonyms etc. (for example, "cola" instead of "Coke"). You can address the robot in different ways. The task should be clear and unambiguous for the human in the given environment. Please, be creative!
  
6. Manually reviewing Mistral's and ChatGPT's answers.

## Full environment list
Non-food items:
1. fridge
2. oven
3. microwave
4. dishwasher
5. sink
6. cutting board
7. soup pot
8. stockpot
9. frying pan
10. grill pan
11.  sauce pan
12.  metal bowl
13.  plastic bowl
14.  ceramic bowl
15.  plastic dinner plate
16.  plastic bread plate
17.  plastic salad plate
18.  ceramic dinner plate
19.  ceramic bread plate
20.  ceramic salad plate
21.  glass dinner plate
22.  glass bread plate
23.  glass salad plate
24.  porcelain cup
25.  beer mug
26.  ceramic mug
27.  glass mug
28.  plastic cup
29.  paper cup
30.  glass
31.  dish soap
32.  paper towels
33.  trash bin
34.  blender
35.  mixer
36.  toaster
37.  coffee machine
38.  bottle opener
39.  tea kettle
40.  corkscrew
41.  whisk
42.  ladle
43.  oven mitts
44.  potholder
45.  kitchen towel
46.  dish rack
47.  vegetable peeler
48.  potato masher
49.  grater
50.  shears
51.  citrus juicer
52.  garlic press
53.  bread knife
54.  paring knife
55.  butter knife
56.  stainless steel tablespoon
57.  wooden tablespoon
58.  silver teaspoon
59.  chopsticks
60.  stainless steel dinner fork
61.  stainless steel salad fork
62.  stainless steel dinner knife
63.  spatula
64.  plastic food storage container 
65.  glass food storage container 
66.  knife block
67.  bottom drawer
68.  middle drawer
69.  top drawer
70.  kitchen table
71.  chair (4)
72.  clean sponge
73.  dirty sponge

Food items:
75.  eggs
76.  olive oil
77.  sunflower oil
78.  coconut oil
79.  sliced whole wheat bread
80.  toasted bread
81.  uncut white bread
82.  a bottle of white wine 
83.  a bottle of red wine 
84.  bottled water
85.  bottled iced tea
86.  beer can
87.  Coka-Cola can
88.  Pepsi can
89.  Sprite bottle
90.  orange soda
91.  RedBull can
92.  rice chips
93.  jalapeno chips
94.  potato chips
95.  energy bar
96.  apple
97.  orange
98.  banana
99.  grapes
100.  lemon
101. avocado
102. peach
103.  table salt
104.  sea salt
105. granulated sugar
106. bell pepper
107. black pepper
108. tomato
109. cucumber
110. potato
111. cabbage
112. carrot
113. onion
114. garlic
115. tomato paste
116. mayonnaise
117. ketchup
118. mustard
119. glass milk bottle
120. oat milk bottle
121. greek yogurt cup
122. vanilla yogurt cup
123. strawberry yogurt cup
124. cheddar cheese slices
125. mozarella sticks
126. fresh mozarella package
127. cream cheese
128. cottage cheese
129. black tea bags
130. green tea bags
131. milk chocolate tablet
132. dark chocolate tablet
133. almond milk chocolate tablet
134. canned olives
135. honey
136. muesli
137. cornflakes
138. mixed fruit jam
139. flour

A fridge, an oven, a kitchen table, a microwave, a dishwasher, a sink and a tea kettle are present in every environment.
