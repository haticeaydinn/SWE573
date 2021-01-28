import tagme
# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "a5a377c1-1bd0-47b9-907a-75b1cdacb1d9-843339462"

lunch_annotations = tagme.annotate("My favourite meal is Mexican burritos.")

# Print annotations with a score higher than 0.1
for ann in lunch_annotations.get_annotations(0.1):
    print(ann)

print("----------------")

lunch_annotations = tagme.annotate("Jeremy Clarkson: Dear the newspapers. I didn’t “battle” Covid. I lay on my bed reading a book till it went away")

# Print annotations with a score higher than 0.1
for ann in lunch_annotations.get_annotations(0.1):
    print(ann, ann.uri())

print("----------------")


tomatoes_mentions = tagme.mentions("I definitely like ice cream better than tomatoes.")

for mention in tomatoes_mentions.mentions:
    print(mention)



print("----------------")



# Get relatedness between a pair of entities specified by title.
rels = tagme.relatedness_title(("Melania Trump","First Lady"))
print(f"Obama and italy have a semantic relation of {rels.relatedness[0].rel}")

# Get relatedness between a pair of entities specified by Wikipedia ID.
rels = tagme.relatedness_wid((31717, 534366))
print(f"IDs 31717 and 534366 have a semantic relation of {rels.relatedness[0].rel}")

# Get relatedness between three pairs of entities specified by title.
# The last entity does not exist, hence the value for that pair will be None.
rels = tagme.relatedness_title([("Barack_Obama", "Italy"),
                                ("Italy", "Germany"),
                                ("Italy", "BAD ENTITY NAME")])
for rel in rels.relatedness:
    print(rel)

# You can also build a dictionary
rels_dict = dict(rels)
print(rels_dict[("Barack Obama", "Italy")])