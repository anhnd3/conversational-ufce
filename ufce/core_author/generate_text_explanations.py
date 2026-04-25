from simplenlg.framework import *
from simplenlg.lexicon import *
from simplenlg.realiser.english import *
from simplenlg.phrasespec import *
from simplenlg.features import *


def features_to_explain(test_df, cfs_df):
    """
    :param test_df: test data
    :param cfs_df: counterfactual data
    :return: feature list, new values to suggest, and actual values to reason
    """
    cfs_df.sort_values(by=cfs_df.columns[0], ascending=True)
    feature_list = []
    new_values = []
    actual_values = []
    for f in cfs_df.columns:
        if cfs_df[f].values != test_df[f].values:
            feature_list.append(f)
            new_values.append(cfs_df[f].values[0])
            actual_values.append(test_df[f].values[0])
    return feature_list, new_values, actual_values

def generate_reason_explanation(outcome_variable, actual_class, features):
    """
    :param outcome_variable:
    :param actual_class:
    :param features:
    :return:
    """
    lexicon = Lexicon.getDefaultLexicon()
    nlgFactory = NLGFactory(lexicon)
    realiser = Realiser(lexicon)
    print("OUTCOME REASONS:")
    # phrase part 1
    subj = outcome_variable  # "loan" could be dynamic
    # subj.setDeterminer("the")
    verb = "be"
    obj = actual_class  # yes or no, need to change dynamically
    phrase_part1 = nlgFactory.createClause()  # "sentence part 1"
    phrase_part1.setSubject(subj)
    phrase_part1.setVerb(verb)
    phrase_part1.setObject(obj)
    # phrase part2
    # subj.setDeterminer("the")
    subject = []  # should be dynamic as per the list of features to change
    if len(features) == 1:
        subj1 = features[0]
    elif len(features) == 2:
        subject1 = nlgFactory.createNounPhrase(features[0])
        subject2 = nlgFactory.createNounPhrase(features[1])
        subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
    elif len(features) == 3:
        subject1 = nlgFactory.createNounPhrase(features[0])
        subject2 = nlgFactory.createNounPhrase(features[1])
        subject3 = nlgFactory.createNounPhrase(features[2])
        subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
        subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
    elif len(features) == 4:
        subject1 = nlgFactory.createNounPhrase(features[0])
        subject2 = nlgFactory.createNounPhrase(features[1])
        subject3 = nlgFactory.createNounPhrase(features[2])
        subject4 = nlgFactory.createNounPhrase(features[3])
        subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
        subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
        subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject4)
    verb1 = "be"
    obj1 = "enough"  # yes or no, need to change dynamically
    phrase_part2 = nlgFactory.createClause()  # "sentence part 1"
    phrase_part2.setSubject(subj1)
    phrase_part2.setVerb(verb1)
    phrase_part2.setObject(obj1)

    phrase_part1.setFeature(Feature.TENSE, Tense.PRESENT)
    phrase_part2.setFeature(Feature.COMPLEMENTISER, "because values of")
    phrase_part2.setFeature(Feature.NEGATED, True)
    phrase_part1.addComplement(phrase_part2)

    output = realiser.realiseSentence(phrase_part1)
    print(output)

def generate_suggestion_explanation(outcome_variable, desired_class, features, new_values):
    """
    :param outcome_variable:
    :param desired_class:
    :param features:
    :param new_values:
    :return:
    """
    lexicon = Lexicon.getDefaultLexicon()
    nlgFactory = NLGFactory(lexicon)
    realiser = Realiser(lexicon)
    print("Suggestion-Explanation:")
    # phrase part 1
    subj = outcome_variable  # "loan" could be dynamic
    verb = "would be"
    obj = desired_class  # yes or no, need to change dynamically
    phrase_part1 = nlgFactory.createClause()  # "sentence part 1"
    phrase_part1.setSubject(subj)
    phrase_part1.setVerb(verb)
    phrase_part1.setObject(obj)
    # phrase part2
    subject = []
    if len(features) == 1:
        subj1 = features[0]
    elif len(features) == 2:
        subject1 = nlgFactory.createNounPhrase(features[0])
        subject2 = nlgFactory.createNounPhrase(features[1])
        subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
    elif len(features) == 3:
        subject1 = nlgFactory.createNounPhrase(features[0])
        subject2 = nlgFactory.createNounPhrase(features[1])
        subject3 = nlgFactory.createNounPhrase(features[2])
        subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
        subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
    elif len(features) == 4:
        subject1 = nlgFactory.createNounPhrase(features[0])
        subject2 = nlgFactory.createNounPhrase(features[1])
        subject3 = nlgFactory.createNounPhrase(features[2])
        subject4 = nlgFactory.createNounPhrase(features[3])
        subj1 = nlgFactory.createCoordinatedPhrase(subject1, subject2)
        subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject3)
        subj1 = nlgFactory.createCoordinatedPhrase(subj1, subject4)
    verb1 = "change features"
    obj1 = "you"  # yes or no, need to change dynamically
    phrase_part2 = nlgFactory.createClause()  # "sentence part 1"
    phrase_part2.setSubject(obj1)
    phrase_part2.setVerb(verb1)
    phrase_part2.setObject(subj1)

    phrase_part1.setFeature(Feature.TENSE, Tense.PRESENT)
    phrase_part2.setFeature(Feature.COMPLEMENTISER, "if")
    phrase_part1.addComplement(phrase_part2)

    # phrase part 3
    if len(features) == 1:
        subj2 = str(new_values[0])
    elif len(features) == 2:
        object1 = nlgFactory.createNounPhrase(str(new_values[0]))
        object2 = nlgFactory.createNounPhrase(str(new_values[1]))
        subj2 = nlgFactory.createCoordinatedPhrase(object1, object2)
    elif len(features) == 3:
        object1 = nlgFactory.createNounPhrase(str(new_values[0]))
        object2 = nlgFactory.createNounPhrase(str(new_values[1]))
        object3 = nlgFactory.createNounPhrase(str(new_values[2]))
        subj2 = nlgFactory.createCoordinatedPhrase(object1, object2)
        subj2 = nlgFactory.createCoordinatedPhrase(subj2, object3)
    elif len(features) == 4:
        object1 = nlgFactory.createNounPhrase(str(new_values[0]))
        object2 = nlgFactory.createNounPhrase(str(new_values[1]))
        object3 = nlgFactory.createNounPhrase(str(new_values[2]))
        object4 = nlgFactory.createNounPhrase(str(new_values[3]))
        subj2 = nlgFactory.createCoordinatedPhrase(object1, object2)
        subj2 = nlgFactory.createCoordinatedPhrase(subj2, object3)
        subj2 = nlgFactory.createCoordinatedPhrase(subj2, object4)
    verb2 = "change features"
    obj2 = "you"  # yes or no, need to change dynamically
    phrase_part3 = nlgFactory.createClause()  # "sentence part 1"
    phrase_part3.setSubject(subj2)

    phrase_part2.setFeature(Feature.TENSE, Tense.PRESENT)
    phrase_part3.setFeature(Feature.COMPLEMENTISER, "to")
    phrase_part1.addComplement(phrase_part3)

    output = realiser.realiseSentence(phrase_part1)
    print(output)
    return output