from sentence_transformers import SentenceTransformer

list1 = [
    "She liked the feeling of the keyboard under her fingers.",
    "Tom's apartment is way too big for just one man.",
    "Lenora only had eight fingers, after losing both of her pinkies to a freak accident with a hay baler.",
    "This ship is too big to pass through the canal.",
    "She wondered how many people actually bought the perch they kept behind the counter at coffee shops.",
    "She grinned with delight at the sight of her ice cream cone.",
    "The tape was so sticky that, when it got stuck to my eyebrows, it wouldn't come off.",
    "In her dreams, she was a Health and Wellness Reporter at the New York Times, but she knew she had a long way to go before she would get there.",
    "Children actors do their best, but they're generally very bad at being convincing.",
    "Let's all just take a moment to breathe, please!"
]

list2 = [
    "The underground bunker was filled with chips and candy.",
    "The spa attendant applied the deep cleaning mask to the gentleman’s back.",
    "He said he was not there yesterday; however, many people saw him there.",
    "The stranger officiates the meal.",
    "I’m a living furnace.",
    "After fighting off the alligator, Brian still had to face the anaconda.",
    "The glacier came alive as the climbers hiked closer.",
    "He liked to play with words in the bathtub.",
    "The sunblock was handed to the girl before practice, but the burned skin was proof she did not apply it.",
    "He fumbled in the darkness looking for the light switch, but when he finally found it there was someone already there."
]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedding1 = model.encode(list1)
embedding2 = model.encode(list2)

print(f"Cosine similarity 1x1:\n {model.similarity_pairwise(embedding1, embedding1)} \n")
print(f"Cosine similarity 1x2:\n {model.similarity_pairwise(embedding1, embedding2)} \n")