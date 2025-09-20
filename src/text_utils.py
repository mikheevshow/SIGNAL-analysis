import re
import logging

logger = logging.getLogger(__name__)

def define_incongruent_word_start_index(congruent_sentence: str, incongruent_sentence: str) -> list[int]:
    split_congruent = congruent_sentence.split()
    split_incongruent = incongruent_sentence.split()
    for w1, w2 in zip(split_congruent, split_incongruent):
        if w1.lower() != w2.lower():
            logger.debug("Searching for incongruent word: {}", w2)
            a = re.search(w2, incongruent_sentence)
            if a is None:
                raise ValueError(f"Sentences: {congruent_sentence}, Sentences: {incongruent_sentence}")
            return [a.start(), a.end()]
    logger.debug("Sentences are equal, both are: {}", congruent_sentence)
    return [-1, -1]


def find_index_of_incongruent_word(congruent_sentences: list[str], incongruent_sentences: list[str]) -> list[list[int]]:
    index_list: list[list[int]] = []
    if len(incongruent_sentences) != len(incongruent_sentences):
        raise ValueError("Incongruent sentences must have same length")
    for congruent_sentence, incongruent_sentence in zip(congruent_sentences, incongruent_sentences):
        i = define_incongruent_word_start_index(congruent_sentence, incongruent_sentence)
        index_list.append(i)
    return index_list



if __name__ == "__main__":

    correct_text = ["Автобусы проходят массовую дезинфекцию"]
    incorrect_text = ["Автобусы проходят массовую дезинфекцией"]

    assert find_index_of_incongruent_word(correct_text, incorrect_text) == [27]

