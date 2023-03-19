import logging
from typing import List, Literal, Optional

import datasets as ds
import mojimoji
from tango import Step
from tango.integrations.datasets import DatasetsFormat

logger = logging.getLogger(__name__)

Analyzer = Literal["jumanpp", "juman", "mecab", "char"]


class Word(object):
    def __init__(self, string: str, pos: Optional[str] = None, offset: int = 0):
        self.string = string
        self.pos = pos
        self.start_position = offset
        self.end_position = offset + len(self.string) - 1


class MorphologicalAnalyzer(object):
    def __init__(
        self,
        analyzer_name: Analyzer,
        mecab_dic_dir: Optional[str] = None,
        is_han_to_zen: bool = False,
    ) -> None:
        self._analyzer_name = analyzer_name
        self._is_han_to_zen = is_han_to_zen

        if self._analyzer_name == "jumanpp":
            from rhoknp import Jumanpp

            self._jumanpp = Jumanpp()

        elif self._analyzer_name == "juman":
            from pyknp import Juman

            self._juman = Juman(jumanpp=False)

        elif self._analyzer_name == "mecab":
            import MeCab

            tagger_option_string = ""
            if mecab_dic_dir is not None:
                tagger_option_string += f" -d {mecab_dic_dir}"
            self._mecab = MeCab.Tagger(tagger_option_string)

        else:
            raise ValueError(f"Invalid analyzer: {self._analyzer_name}")

    def _get_words_jumanpp(self, string: str) -> List[Word]:
        words = []
        offset = 0

        try:
            result = self._jumanpp.apply_to_sentence(string)
        except ValueError as err:
            logger.warning(f"{err}. skip sentence: {string}")

        for mrph in result.morphemes:
            words.append(Word(mrph.text, pos=mrph.pos, offset=offset))
            offset += len(mrph.text)

        return words

    def _get_words_juman(self, string: str) -> List[Word]:
        words = []
        offset = 0

        try:
            result = self._juman.analysis(string)
        except ValueError as err:
            logger.warning(f"{err}. skip sentence: {string}")
            return []

        for mrph in result.mrph_list():
            words.append(Word(mrph.midasi, pos=mrph.hinsi, offset=offset))
            offset += len(mrph.midasi)
        return words

    def _get_words_mecab(self, string: str) -> List[Word]:
        words = []
        offset = 0

        self._mecab.parse("")
        node = self._mecab.parseToNode(string)
        while node:
            word = node.surface
            pos = node.feature.split(",")[0]
            if node.feature.split(",")[0] != "BOS/EOS":
                words.append(Word(word, pos=pos, offset=offset))
                offset += len(word)
            node = node.next
        return words

    def _get_words_char(self, string: str) -> List[Word]:
        words = []
        offset = 0

        for char in list(string):
            words.append(Word(char, offset=offset))
            offset += 1
        return words

    def get_words(self, string: str) -> List[Word]:
        if self._analyzer_name == "jumanpp":
            words = self._get_words_jumanpp(string)
        elif self._analyzer_name == "juman":
            words = self._get_words_juman(string)
        elif self._analyzer_name == "mecab":
            words = self._get_words_mecab(string)
        elif self._analyzer_name == "char":
            words = self._get_words_char(string)
        else:
            raise ValueError(f"Invalid analyzer: {self._analyzer_name}")

        return words

    def get_tokenized_string(self, string: str) -> Optional[str]:
        if self._is_han_to_zen:
            string = mojimoji.han_to_zen(string)

        words = self.get_words(string)
        if len(words) > 0:
            return " ".join([word.string for word in words])
        else:
            return None


@Step.register("apply_morphological_analysis")
class ApplyMorphologicalAnalysis(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT = DatasetsFormat()

    def run(  # type: ignore[override]
        self,
        dataset: ds.DatasetDict,
        analyzer_name: Analyzer,
        column_names: List[str],
        mecab_dic_dir: Optional[str] = None,
        is_han_to_zen: bool = False,
    ) -> ds.DatasetInfo:
        analyzer = MorphologicalAnalyzer(
            analyzer_name=analyzer_name,
            mecab_dic_dir=mecab_dic_dir,
            is_han_to_zen=is_han_to_zen,
        )

        def preprocess_function(example):
            for column in column_names:
                string = example[column]
                tokenized_string = analyzer.get_tokenized_string(string)
                example[column] = tokenized_string
            return example

        dataset = dataset.map(
            function=preprocess_function,
            desc="Applying morphological analysis",
        )

        return dataset
